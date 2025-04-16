#slack10_re.py
import asyncio
import sys
import os
import json
import threading
import time
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import AsyncExitStack
import glob
import re

from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class ConversationMemory:
    """A class to store conversation history and context for each Slack channel/thread"""
    
    def __init__(self):
        self.conversations = {}  # Dict to store conversations by channel_id or thread_id
        self.context = {}  # Dict to store additional context by channel_id or thread_id
        self.session_context = {}  # Store persistent context for each session
        self.expiry_time = 60 * 60  # 1 hour expiry by default
        
    def get_conversation(self, session_id: str) -> List:
        """Get conversation history for a session"""
        # Remove expired conversations first
        self._clean_expired_sessions()
        
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'last_updated': time.time()
            }
        return self.conversations[session_id]['messages']
    
    def add_message(self, session_id: str, message: Dict) -> None:
        """Add a message to the conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'last_updated': time.time()
            }
        
        self.conversations[session_id]['messages'].append(message)
        self.conversations[session_id]['last_updated'] = time.time()
    
    def store_context(self, session_id: str, key: str, value) -> None:
        """Store context data for a session"""
        if session_id not in self.context:
            self.context[session_id] = {}
        
        self.context[session_id][key] = value
        
    def get_context(self, session_id: str, key: str, default=None):
        """Get context data for a session"""
        if session_id not in self.context:
            return default
        return self.context[session_id].get(key, default)
    
    def set_session_context(self, session_id: str, context_data: Dict) -> None:
        """Store persistent context information for a session that should be included in every prompt"""
        if session_id not in self.session_context:
            self.session_context[session_id] = {}
        
        # Update with new context data
        self.session_context[session_id].update(context_data)
    
    def get_session_context(self, session_id: str) -> Dict:
        """Get the persistent context data for a session"""
        if session_id not in self.session_context:
            return {}
        return self.session_context[session_id]
    
    def clear_session_context(self, session_id: str) -> None:
        """Clear the persistent context for a session"""
        if session_id in self.session_context:
            del self.session_context[session_id]
    
    def get_enhanced_system_prompt(self, session_id: str, base_prompt: str) -> str:
        """Enhance a system prompt with session context information"""
        context = self.get_session_context(session_id)
        if not context:
            return base_prompt
        
        # Format context information
        context_section = "CURRENT CONTEXT:\n"
        for key, value in context.items():
            if isinstance(value, dict):
                context_section += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    context_section += f"  - {sub_key}: {sub_value}\n"
            else:
                context_section += f"- {key}: {value}\n"
        
        # Combine with base prompt
        enhanced_prompt = f"{base_prompt}\n\n{context_section}\n\nWhen using tools that require user information, use the user ID from CURRENT CONTEXT. Do not ask the user for their user_id as you already have this information."
        return enhanced_prompt
    
    def _clean_expired_sessions(self) -> None:
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, data in self.conversations.items():
            if current_time - data['last_updated'] > self.expiry_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if session_id in self.conversations:
                del self.conversations[session_id]
            if session_id in self.context:
                del self.context[session_id]
            if session_id in self.session_context:
                del self.session_context[session_id]


class MCPClient:
    def __init__(self, server_paths: List[str] = None):
        # Initialize session and client objects
        self.sessions = {}  # Dictionary to store multiple sessions
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.memory = ConversationMemory()  # Add conversation memory
        
        # If no server paths provided, use default paths
        self.server_paths = server_paths or []
        if not self.server_paths:
            # Auto-discover server scripts in the 'server' directory
            if os.path.exists('server'):
                self.server_paths.extend(glob.glob('server/*.py'))
                self.server_paths.extend(glob.glob('server/*.js'))

    def parse_tool_result(self, result_content) -> Dict:
        """Parse the result content from a tool call, handling different formats"""
        try:
            if isinstance(result_content, list) and len(result_content) > 0 and hasattr(result_content[0], 'text'):
                # If it's a list of TextContent objects, extract the text from the first item
                result_text = result_content[0].text
                return json.loads(result_text)
            elif isinstance(result_content, dict):
                return result_content
            elif isinstance(result_content, str):
                return json.loads(result_content)
            else:
                print(f"Warning: Unexpected result type: {type(result_content)}")
                # If we can't parse it as a dictionary, return an empty one
                return {"success": False, "error": f"Unexpected result type: {type(result_content)}"}
        except Exception as e:
            print(f"Error parsing tool result: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return {"success": False, "error": f"Failed to parse result: {str(e)}"}

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()

        # Store session with server path as key
        self.sessions[server_script_path] = {
            'session': session,
            'stdio': stdio,
            'write': write
        }

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to server {server_script_path} with tools:", [tool.name for tool in tools])

    async def register_user_immediately(self, user_id: str, username: str) -> bool:
        """Directly register a user in the database without any checks"""
        try:
            # Find the review server
            review_server = None
            for server_id, server_data in self.sessions.items():
                if "review" in server_id.lower():
                    review_server = server_data['session']
                    break
            
            if not review_server:
                print(f"Warning: Could not find review server when registering user {username}")
                return False
            
            # Directly call register_user
            reg_result = await review_server.call_tool("register_user", {
                "user_id": user_id, 
                "username": username
            })
            
            # Parse result using helper function
            reg_data = self.parse_tool_result(reg_result.content)
            
            if reg_data.get("success"):
                print(f"Successfully registered user: {username} ({user_id})")
                return True
            else:
                print(f"Failed to register user: {reg_data.get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"Error in register_user_immediately: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return False

    async def upload_file_to_slack(self, channel_id: str, file_path: str, title: str = None, comment: str = None):
        """Upload a file to Slack"""
        try:
            # Initialize Slack client
            slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            
            print(f"Uploading file to channel ID: {channel_id}")
            file_name = os.path.basename(file_path)
            
            # Get file size in bytes
            file_size = os.path.getsize(file_path)
            
            # Step 1: Get upload URL from Slack
            upload_url_response = slack_client.files_getUploadURLExternal(
                filename=file_name,
                length=file_size
            )
            
            if not upload_url_response["ok"]:
                print(f"Failed to get upload URL: {upload_url_response.get('error', 'Unknown error')}")
                return None
            
            upload_url = upload_url_response["upload_url"]
            file_id = upload_url_response["file_id"]
            
            print(f"Got upload URL and file ID: {file_id}")
            
            # Step 2: Upload file to the provided URL
            with open(file_path, 'rb') as file_content:
                # Using requests library to POST the file
                import requests
                
                headers = {
                    'Content-Type': 'application/octet-stream',
                }
                
                upload_response = requests.post(
                    upload_url,
                    headers=headers,
                    data=file_content
                )
                
                if upload_response.status_code != 200:
                    print(f"Failed to upload file: HTTP {upload_response.status_code}, {upload_response.text}")
                    return None
                    
                print(f"File uploaded successfully to temporary URL")
            
            # Step 3: Complete the upload process to finalize and share
            complete_args = {
                "files": [{"id": file_id, "title": title or file_name}]
            }
            
            # Add channel information for sharing
            if channel_id:
                complete_args["channel_id"] = channel_id
            
            # Add initial comment if provided
            if comment:
                complete_args["initial_comment"] = comment
                
            complete_response = slack_client.files_completeUploadExternal(**complete_args)
            
            if not complete_response["ok"]:
                print(f"Failed to complete upload: {complete_response.get('error', 'Unknown error')}")
                return None
                
            print(f"Upload completed and shared with result: {complete_response}")
            return complete_response
                    
        except Exception as e:
            print(f"Error uploading file to Slack: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return None
    async def handle_pdf_generation_result(self, channel_id: str, result: Any, user_id: str = None, invoice_data: dict = None, html_content: str = None):
        """Handle the result of PDF generation by uploading to Slack and optionally saving invoice data
        
        Args:
            channel_id: The Slack channel ID
            result: The result from the PDF generation tool
            user_id: The Slack user ID (optional)
            invoice_data: Dictionary with invoice metadata (optional)
            html_content: The HTML content that was converted to PDF (optional)
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Add these debug lines right at the beginning
            print(f"===PDF RESULT RAW===: {result}")
            
            # Parse the result using helper function
            result_data = self.parse_tool_result(result)
            print(f"===PDF RESULT PARSED===: {json.dumps(result_data, indent=2)}")
            
            # Check if PDF generation was successful
            if result_data.get("status") != "success" or "filepath" not in result_data:
                print(f"===PDF ERROR===: Missing success status or filepath")
                return False, f"PDF generation failed: {result_data.get('message', 'Unknown error')}"
            
            file_path = result_data["filepath"]
            print(f"===PDF FILEPATH===: {file_path}")
            print(f"===FILE EXISTS===: {os.path.exists(file_path)}")
            
            # If this is an invoice and we have the HTML content, save it to the database
            if invoice_data and html_content and user_id:
                save_result = await self.save_generated_invoice(
                    html_content=html_content,
                    invoice_data=invoice_data,
                    pdf_path=file_path,
                    creator_id=user_id
                )
                
                if save_result.get("success"):
                    print(f"===INVOICE SAVED===: ID: {save_result.get('invoice_id')}")
                else:
                    print(f"===INVOICE SAVE ERROR===: {save_result.get('error')}")
            
            # Prepare comment for the file upload
            comment = f"Here's your requested PDF file"
            if user_id:
                comment = f"<@{user_id}>: {comment}"
            
            # Upload the file to Slack
            file_name = result_data.get("filename", os.path.basename(file_path))
            
            # Upload the PDF
            upload_result = await self.upload_file_to_slack(
                channel_id=channel_id,
                file_path=file_path,
                title=file_name,
                comment=comment
            )
            
            # Add more debug code here
            print(f"===UPLOAD RESULT===: {upload_result}")
            
            if not upload_result:
                return False, "Failed to upload the PDF file to Slack"
            
            # Get file ID and permalink from upload result
            file_id = None
            permalink = None
            
            if upload_result and upload_result.get("files") and len(upload_result["files"]) > 0:
                file_id = upload_result["files"][0].get("id")
                
                # Get file info to retrieve permalink
                if file_id:
                    slack_client_obj = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
                    file_info = slack_client_obj.files_info(file=file_id)
                    if file_info and file_info.get("ok"):
                        permalink = file_info.get("file", {}).get("permalink")
            
            # Ask if the user wants to create a review for this file
            question_msg = f"Would you like to create a review for this file? Reply with 'create-review TITLE [DESCRIPTION]' to create a review."
            
            slack_client_obj = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            slack_client_obj.chat_postMessage(
                channel=channel_id,
                text=question_msg
            )
            
            # Store the file info in context for later use - now including Slack file ID and permalink
            self.memory.store_context(channel_id, "last_generated_file", {
                "path": file_path,
                "name": file_name,
                "type": "pdf",
                "slack_file_id": file_id,
                "slack_permalink": permalink
            })
            
            # Success message
            return True, f"Successfully uploaded PDF file: {file_name}"
            
        except Exception as e:
            print(f"===PDF EXCEPTION===: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"===TRACEBACK===: {traceback_info}")
            return False, f"Error processing PDF: {str(e)}"
    async def process_invoice_request(self, channel_id: str, invoice_id: int, user_id: str = None):
        """Process a request to fetch and convert an invoice to PDF"""
        try:
            # Find the review server to get the invoice
            review_server = None
            for server_id, server_data in self.sessions.items():
                if "review" in server_id.lower():
                    review_server = server_data['session']
                    break
            
            if not review_server:
                return False, "Could not find review server"
            
            # Get the invoice
            result = await review_server.call_tool("get_invoice_by_id", {"invoice_id": invoice_id})
            
            # Parse result
            result_data = self.parse_tool_result(result.content)
            if not result_data.get("success"):
                return False, f"Failed to retrieve invoice: {result_data.get('error')}"
            
            invoice = result_data.get("invoice", {})
            html_content = invoice.get("html_content")
            invoice_number = invoice.get("invoice_number", "Unknown")
            client_name = invoice.get("client_name", "Unknown Client")
            title = f"Invoice {invoice_number} - {client_name}"
            
            # Find the PDF server to convert the HTML
            pdf_server = None
            for server_id, server_data in self.sessions.items():
                if "pdf" in server_id.lower():
                    pdf_server = server_data['session']
                    break
            
            if not pdf_server:
                return False, "Could not find PDF converter server"
            
            # Convert to PDF
            pdf_result = await pdf_server.call_tool("convert_html_to_pdf", {
                "html_content": html_content,
                "title": title
            })
            
            # Handle the PDF generation result
            success, message = await self.handle_pdf_generation_result(
                channel_id=channel_id,
                result=pdf_result.content,
                user_id=user_id
            )
            
            # If the PDF generation was successful, update the invoice record with the PDF path
            if success and pdf_result.content:
                pdf_data = self.parse_tool_result(pdf_result.content)
                if pdf_data.get("status") == "success" and pdf_data.get("filepath"):
                    await review_server.call_tool("update_invoice_pdf_path", {
                        "invoice_id": invoice_id,
                        "pdf_path": pdf_data.get("filepath")
                    })
            
            return success, message
            
        except Exception as e:
            print(f"Error processing invoice: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return False, f"Error processing invoice: {str(e)}"
    
    async def save_generated_invoice(self, html_content: str, invoice_data: dict, pdf_path: str, creator_id: str):
        """Save a generated invoice to the database
        
        Args:
            html_content: The full HTML content of the invoice
            invoice_data: Dictionary with invoice metadata (invoice_number, client_name, etc.)
            pdf_path: Path to the generated PDF file
            creator_id: The Slack user ID of the creator
            
        Returns:
            Dict with status information
        """
        try:
            # Find the review server to save the invoice
            review_server = None
            for server_id, server_data in self.sessions.items():
                if "review" in server_id.lower():
                    review_server = server_data['session']
                    break
            
            if not review_server:
                return {"success": False, "error": "Could not find review server"}
            
            # Extract invoice data
            invoice_number = invoice_data.get("invoice_number", f"INV-{int(time.time())}")
            client_name = invoice_data.get("client_name", "")
            client_email = invoice_data.get("client_email", "")
            amount = invoice_data.get("amount", 0.0)
            currency = invoice_data.get("currency", "USD")
            description = invoice_data.get("description", "")
            
            # Save the invoice to the database
            result = await review_server.call_tool("save_invoice", {
                "invoice_number": invoice_number,
                "client_name": client_name,
                "client_email": client_email,
                "amount": amount,
                "currency": currency,
                "description": description,
                "html_content": html_content,
                "pdf_path": pdf_path,
                "creator_id": creator_id
            })
            
            return self.parse_tool_result(result.content)
            
        except Exception as e:
            print(f"Error saving invoice: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return {"success": False, "error": f"Error saving invoice: {str(e)}"}
    
    async def notify_user_of_assignment(self, user_id, review_id, review_title, assigner_id):
        """Notify a user that they've been assigned a review"""
        try:
            slack_client_obj = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            
            # Get assigner's name
            assigner_info = slack_client_obj.users_info(user=assigner_id)
            assigner_name = assigner_info["user"]["name"]
            
            # Find the review server for details
            review_server = None
            for server_id, server_data in self.sessions.items():
                if "review" in server_id.lower():
                    review_server = server_data['session']
                    break
            
            if review_server:
                # Get review details to get the title
                result = await review_server.call_tool("get_review_details", {"review_id": review_id})
                
                # Parse result using helper function
                result_data = self.parse_tool_result(result.content)
                    
                if result_data.get("success"):
                    review_title = result_data.get("review", {}).get("title", "Review")
            
            # Send a DM to the user
            slack_client_obj.chat_postMessage(
                channel=user_id,  # In Slack, sending a message to a user ID opens a DM
                text=f"You've been assigned a new review by {assigner_name}:\n*{review_title}* (ID: {review_id})\n\nUse 'check-inbox' in any channel where the bot is present to see your pending reviews."
            )
            
            return True
        except Exception as e:
            print(f"Error notifying user: {str(e)}")
            return False

    async def notify_user_of_comment(self, user_id, review_id, review_title, commenter_id, commenter_name, comment_text):
        """Notify a user that a new comment was added to a review they're assigned to"""
        try:
            slack_client_obj = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            
            # Truncate comment if too long
            short_comment = comment_text[:100] + "..." if len(comment_text) > 100 else comment_text
            
            # Send a DM to the user
            slack_client_obj.chat_postMessage(
                channel=user_id,
                text=f"New comment from {commenter_name} on review *{review_title}* (ID: {review_id}):\n\n>{short_comment}\n\nUse 'review-details {review_id}' to see all comments."
            )
            
            return True
        except Exception as e:
            print(f"Error notifying user of comment: {str(e)}")
            return False

    async def process_query(self, query: str, channel_id: str, user_id: str = None, username: str = None) -> str:
        """Process a query using Claude and available tools"""
        # Get the conversation history for this channel
        conversation = self.memory.get_conversation(channel_id)
        
        # Store user context for this session
        if user_id and username:
            self.memory.set_session_context(channel_id, {
                "user": {
                    "id": user_id,
                    "username": username
                }
            })
        
        # Add current query to conversation
        conversation.append({
            "role": "user",
            "content": query
        })
        
        # Create a copy of the conversation for this API call
        messages = conversation.copy()

        # Collect tools from all connected servers
        available_tools = []
        for server_path, server_data in self.sessions.items():
            # Create a clean server ID by replacing invalid characters
            server_id = server_path.replace('/', '_').replace('.', '_').replace('\\', '_')
            response = await server_data['session'].list_tools()
            for tool in response.tools:
                available_tools.append({
                    "name": f"{server_id}__{tool.name}",  # Use double underscore as separator
                    "description": f"[{server_path}] {tool.description}",  # Add server path to description
                    "input_schema": tool.inputSchema
                })

        # For tracking task completion
        is_task_complete = False
        max_iterations = 5  # Safety limit to prevent infinite loops
        iterations = 0
        final_text = []
        
        # Create a comprehensive system prompt that explains available commands and functionality
        base_system_prompt = """You are a helpful assistant integrated with a review management system in Slack. 
        You have access to various tools to help users manage document reviews, create PDF files, collaborate with team members, and manage invoices.

        AVAILABLE FUNCTIONALITY:
        - Creating and managing document reviews
        - Assigning reviews to team members
        - Commenting on reviews
        - Checking your inbox for pending reviews
        - Viewing review details and associated files
        - Generating PDF documents
        - Storing and retrieving invoices

        You can understand user intent and select the appropriate tools to help them. If someone asks to "show my reviews" or "what reviews are assigned to me", 
        you should use the get_inbox tool to help them. If they want to see details about a specific review, use get_review_details.

        If a user asks to view, display or access a file from a review, you should use the appropriate tools to help them access that file.
        
        if a user query about existing invoices, you should use the search_invoices tool to help them.

        When using tools that require user information, use the user_id provided in the CURRENT CONTEXT section.
        Do not ask the user for their user_id or username as you already have this information.

        When you complete a multi-step task, end your response with '[TASK COMPLETED]'.
        If you need to perform additional steps, end your response with '[NEXT STEP]'.
        """
        
        # Get enhanced system prompt with context
        system_prompt = self.memory.get_enhanced_system_prompt(channel_id, base_system_prompt)
        
        # Track notifications needed after tool calls
        notifications_needed = []
        
        # Main processing loop - continues until task is complete or max iterations reached
        while not is_task_complete and iterations < max_iterations:
            iterations += 1
            
            # Make Claude API call
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=available_tools
            )

            # Storage for the assistant's response parts
            assistant_message_content = []
            response_text = ""
            requires_tool_call = False
            
            # Process each part of the response
            for content in response.content:
                if content.type == 'text':
                    response_text += content.text
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                    
                elif content.type == 'tool_use':
                    requires_tool_call = True
                    # Parse server ID and tool name
                    server_id, tool_name = content.name.split('__', 1)
                    
                    # Find the original server path
                    server_path = next(
                        path for path in self.sessions.keys() 
                        if server_id == path.replace('/', '_').replace('.', '_').replace('\\', '_')
                    )
                    tool_args = content.input
                    
                    # If tool needs user_id and it's not provided, add it from context
                    if user_id and "user_id" in tool_args and not tool_args.get("user_id"):
                        tool_args["user_id"] = user_id
                        print(f"Added user_id {user_id} to tool args for {tool_name}")
                        
                    # Execute tool call on the appropriate server
                    result = await self.sessions[server_path]['session'].call_tool(tool_name, tool_args)
                    
                    # Add formatted tool call to final text
                    tool_call_text = f"[Calling tool {tool_name} on {server_path} with args {json.dumps(tool_args)}]"
                    final_text.append(tool_call_text)
                    
                    # Process specific post-tool actions based on the tool used
                    
                    # Handle PDF generation
                    if tool_name in ["json_to_pdf", "convert_html_to_pdf"]:
                        success, message = await self.handle_pdf_generation_result(
                            channel_id=channel_id,
                            result=result.content,
                            user_id=user_id
                        )
                        final_text.append(f"\n\n{message}")
                    
                    # Check for notification-requiring actions
                    result_data = self.parse_tool_result(result.content)
                    
                    # Handle notifications for assign_review
                    if tool_name == "assign_review" and result_data.get("success"):
                        review_id = tool_args.get("review_id")
                        assignee_id = tool_args.get("assignee_id")
                        if review_id and assignee_id:
                            notifications_needed.append({
                                "type": "assignment",
                                "assignee_id": assignee_id,
                                "review_id": review_id,
                                "title": "Review",
                                "assigner_id": user_id
                            })
                    
                    # Handle notifications for add_comment
                    if tool_name == "add_comment" and result_data.get("success"):
                        review_id = tool_args.get("review_id")
                        comment_text = tool_args.get("content", "")
                        
                        # Get review details to find assignees
                        for server_id, server_data in self.sessions.items():
                            if "review" in server_id.lower():
                                review_server = server_data['session']
                                details_result = await review_server.call_tool("get_review_details", {"review_id": review_id})
                                details_data = self.parse_tool_result(details_result.content)
                                
                                if details_data.get("success"):
                                    review_title = details_data.get("review", {}).get("title", "Review")
                                    assignments = details_data.get("assignments", [])
                                    
                                    # Get commenter username
                                    slack_client_obj = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
                                    user_info = slack_client_obj.users_info(user=user_id)
                                    username = user_info["user"]["name"]
                                    
                                    # Add notifications for each assignee
                                    for assignment in assignments:
                                        assignee_id = assignment.get("assignee_id")
                                        if assignee_id and assignee_id != user_id:  # Don't notify the commenter
                                            notifications_needed.append({
                                                "type": "comment",
                                                "assignee_id": assignee_id,
                                                "review_id": review_id,
                                                "title": review_title,
                                                "commenter_id": user_id,
                                                "commenter_name": username,
                                                "comment_text": comment_text
                                            })
                    
                    # Handle file viewing for get_review_details
                    if tool_name == "get_review_details" and result_data.get("success"):
                        review = result_data.get("review", {})
                        
                        # If there's a file path but no Slack file ID or permalink
                        if review.get('file_path') and not (review.get('slack_file_id') and review.get('slack_permalink')):
                            file_path = review.get('file_path')
                            if os.path.exists(file_path):
                                file_name = os.path.basename(file_path)
                                review_id = review.get('id')
                                
                                # Upload the file to Slack
                                upload_result = await self.upload_file_to_slack(
                                    channel_id=channel_id,
                                    file_path=file_path,
                                    title=f"Review #{review_id}: {file_name}",
                                    comment=f"Here's the file for review #{review_id}: {review['title']}"
                                )
                                
                                # If upload successful, update the review record
                                if upload_result and upload_result.get("ok"):
                                    file_id = None
                                    permalink = None
                                    
                                    if upload_result.get("files") and len(upload_result["files"]) > 0:
                                        file_id = upload_result["files"][0].get("id")
                                        
                                        # Get file info to retrieve permalink
                                        if file_id:
                                            slack_client_obj = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
                                            file_info = slack_client_obj.files_info(file=file_id)
                                            if file_info and file_info.get("ok"):
                                                permalink = file_info.get("file", {}).get("permalink")
                                    
                                    # Update the review record with file info
                                    if file_id and permalink:
                                        # Find the review server
                                        for srv_id, srv_data in self.sessions.items():
                                            if "review" in srv_id.lower():
                                                rev_server = srv_data['session']
                                                await rev_server.call_tool("update_review_file_info", {
                                                    "review_id": review_id,
                                                    "slack_file_id": file_id,
                                                    "slack_permalink": permalink
                                                })
                                
                                final_text.append(f"\n\nI've uploaded the associated file for this review to make it easier to access.")
                                
                    # Handle invoice search results - auto-generate PDF when invoice is found
                    if tool_name in ["search_invoices", "get_invoice_by_id", "get_invoice_by_number"] and result_data.get("success"):
                        # For search_invoices, we need to get the first result and then get that invoice
                        if tool_name == "search_invoices" and result_data.get("results") and len(result_data.get("results", [])) > 0:
                            invoice_id = result_data["results"][0]["id"]
                            
                            # Find the review server to get the full invoice
                            review_server = None
                            for srv_id, srv_data in self.sessions.items():
                                if "review" in srv_id.lower():
                                    review_server = srv_data['session']
                                    break
                                    
                            if review_server:
                                # Get the full invoice details
                                invoice_result = await review_server.call_tool("get_invoice_by_id", {"invoice_id": invoice_id})
                                invoice_data = self.parse_tool_result(invoice_result.content)
                                
                                if invoice_data.get("success") and invoice_data.get("invoice", {}).get("html_content"):
                                    # Process invoice request to generate PDF
                                    success, message = await self.process_invoice_request(
                                        channel_id=channel_id, 
                                        invoice_id=invoice_id,
                                        user_id=user_id
                                    )
                                    
                                    if success:
                                        final_text.append(f"\n\nI've automatically generated a PDF of this invoice for you.")
                        
                        # For direct invoice retrieval, we can use the invoice directly
                        elif tool_name in ["get_invoice_by_id", "get_invoice_by_number"] and result_data.get("invoice", {}).get("html_content"):
                            invoice_id = result_data["invoice"]["id"]
                            
                            # Process invoice request to generate PDF
                            success, message = await self.process_invoice_request(
                                channel_id=channel_id, 
                                invoice_id=invoice_id,
                                user_id=user_id
                            )
                            
                            if success:
                                final_text.append(f"\n\nI've automatically generated a PDF of this invoice for you.")

                    # Add tool call to assistant message content
                    assistant_message_content.append(content)
                    
                    # Add the assistant's response (with tool call) to the conversation history
                    self.memory.add_message(channel_id, {
                        "role": "assistant",
                        "content": assistant_message_content.copy()  # Important to copy
                    })
                    
                    # Add the tool result to the conversation history
                    self.memory.add_message(channel_id, {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })
                    
                    # Store key results in context memory for future reference
                    try:
                        context_key = f"{tool_name}_result"
                        self.memory.store_context(channel_id, context_key, result_data)
                    except Exception as e:
                        print(f"Error storing result in context: {str(e)}")
                    
                    # Get updated messages for the next API call
                    messages = self.memory.get_conversation(channel_id)
                    
                    # Filter out empty messages which cause API errors
                    filtered_messages = []
                    for msg in messages:
                        # Check if content is non-empty before adding to filtered messages
                        if isinstance(msg.get('content'), str) and msg['content'].strip():
                            filtered_messages.append(msg)
                        elif isinstance(msg.get('content'), list) and len(msg['content']) > 0:
                            # For list content (like tool results), verify they're not empty
                            if all(item is not None for item in msg['content']):
                                filtered_messages.append(msg)
                    
                    # Update messages for the next iteration
                    messages = filtered_messages
                    
                    # Reset for next tool call processing
                    assistant_message_content = []
                    break  # Process one tool call at a time
            
            # Determine if the task is complete
            if "[TASK COMPLETED]" in response_text:
                is_task_complete = True
            elif "[NEXT STEP]" in response_text:
                is_task_complete = False
            elif not requires_tool_call:
                is_task_complete = True
        
        # Add the final assistant response to conversation history if there's content
        if assistant_message_content:
            self.memory.add_message(channel_id, {
                "role": "assistant",
                "content": assistant_message_content
            })
        
        # Process any needed notifications after the conversation
        for notification in notifications_needed:
            if notification["type"] == "assignment":
                await self.notify_user_of_assignment(
                    notification["assignee_id"],
                    notification["review_id"],
                    notification["title"],
                    notification["assigner_id"]
                )
            elif notification["type"] == "comment":
                await self.notify_user_of_comment(
                    notification["assignee_id"],
                    notification["review_id"],
                    notification["title"],
                    notification["commenter_id"],
                    notification["commenter_name"],
                    notification["comment_text"]
                )
        
        return "\n".join(final_text)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def initialize_servers(self):
        """Initialize all server connections"""
        if not self.server_paths:
            print("No server scripts found. Use 'add-server <path>' to add a server.")
            return
            
        for server_path in self.server_paths:
            try:
                await self.connect_to_server(server_path)
            except Exception as e:
                print(f"Failed to connect to server {server_path}: {str(e)}")


# Initialize Flask app
app = Flask(__name__)

# Initialize MCP client with default paths
def initialize_slack_bot(server_paths=None):
    # Initialize Slack client
    slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
    
    # Initialize MCP client
    mcp_client = MCPClient(server_paths)
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Set up the MCP client
    async def setup_mcp():
        await mcp_client.initialize_servers()
        print("MCP client initialized and connected to servers")
    
    # Run the setup
    loop.run_until_complete(setup_mcp())
    
    # Process a message and send a response
    def process_and_respond(user_query, channel, user_id, thread_ts=None):
        """Process a message and send a response to Slack"""
        try:
            # Trim the message to avoid issues with extra spaces
            user_query = user_query.strip()
            
            # Get username from Slack API
            user_info = slack_client.users_info(user=user_id)
            username = user_info["user"]["name"]
            
            # Use thread_ts if available, otherwise use channel as the conversation ID
            session_id = thread_ts if thread_ts else channel
            
            # Always register the user immediately at the start of every interaction
            try:
                loop.run_until_complete(mcp_client.register_user_immediately(user_id, username))
            except Exception as e:
                print(f"Error registering user: {e}")
            
            # Special case for user ID query - handle this outside of Claude
            if user_query.lower() in ["what is my user id", "what's my user id", "what is my user id?", "what's my user id?", "check my user info", "check my user information"]:
                response = f"Your Slack user ID is: `{user_id}`\nYour username is: `{username}`\nYou are registered in the system."
                slack_client.chat_postMessage(
                    channel=channel,
                    text=f"<@{user_id}>: {response}",
                    thread_ts=thread_ts
                )
                return
            
            # Get initial response message to update with progress
            initial_response = slack_client.chat_postMessage(
                channel=channel,
                text=f"<@{user_id}>: Processing your request...",
                thread_ts=thread_ts
            )
            message_ts = initial_response["ts"]
            
            # Process the query through MCP client - pass both user_id and username
            response = loop.run_until_complete(
                mcp_client.process_query(user_query, session_id, user_id, username)
            )
            
            # Update the initial message with the full response
            slack_client.chat_update(
                channel=channel,
                ts=message_ts,
                text=f"<@{user_id}>: {response}"
            )
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            slack_client.chat_postMessage(
                channel=channel,
                text=f"Sorry <@{user_id}>, I encountered an error trying to process your request: {str(e)}",
                thread_ts=thread_ts
            )
    
    # Handle Slack events
    @app.route("/slack/events", methods=["POST"])
    def slack_events():
        # Get the event data
        data = request.json
        print(f"Received event: {json.dumps(data, indent=2)}")
        
        # Handle URL verification (required when setting up Events API)
        if data.get("type") == "url_verification":
            return jsonify({"challenge": data.get("challenge")})
        
        # Handle events
        if data.get("type") == "event_callback":
            event = data.get("event", {})
            
            # Handle message events
            if event.get("type") == "message":
                # Ignore bot messages and message_changed events to prevent loops
                if "bot_id" not in event and event.get("subtype") != "message_changed":
                    user_query = event.get("text", "")
                    channel = event.get("channel")
                    user_id = event.get("user")
                    thread_ts = event.get("thread_ts")  # Get thread timestamp if in a thread
                    
                    # Only process if we have a valid user_id to prevent None user issues
                    if user_id:
                        print(f"Received message from {user_id}: {user_query}")
                        
                        try:
                            # Try to pre-register user immediately
                            # Get username from Slack API
                            user_info = slack_client.users_info(user=user_id)
                            username = user_info["user"]["name"]
                            
                            # Run this in the main event loop to avoid threading issues
                            loop.run_until_complete(mcp_client.register_user_immediately(user_id, username))
                        except Exception as e:
                            print(f"Error pre-registering user: {e}")
                        
                        # Process the message in a separate thread to avoid blocking
                        thread = threading.Thread(
                            target=process_and_respond,
                            args=(user_query, channel, user_id, thread_ts)
                        )
                        thread.start()
                    else:
                        print(f"Skipping message with missing user_id: {event}")
                elif "bot_id" in event:
                    print(f"Ignoring bot message: {event.get('text', '')[:30]}...")
                elif event.get("subtype") == "message_changed":
                    print(f"Ignoring message_changed event")
            
            # Handle file_shared events (future enhancement)
            elif event.get("type") == "file_shared":
                # Placeholder for file handling logic
                pass
        
        # Return a 200 response to acknowledge receipt of the event
        return "", 200
    
    # Health check endpoint
    @app.route("/", methods=["GET"])
    def health():
        return "MCP Slack Bot is running!"
    
    return loop, mcp_client


def main():
    # Extract command-line arguments, if any
    server_paths = sys.argv[1:] if len(sys.argv) > 1 else None
    
    print("Starting MCP Slack Bot...")
    if server_paths:
        print(f"Server paths: {server_paths}")
    else:
        print("No server paths provided. Auto-discovering servers in 'server' directory.")
    
    # Initialize the slack bot with the provided server paths
    loop, mcp_client = initialize_slack_bot(server_paths)
    
    try:
        # Run the Flask app
        app.run(port=3000)
    except KeyboardInterrupt:
        print("Shutting down MCP Slack Bot...")
    finally:
        # Clean up MCP client resources
        loop.run_until_complete(mcp_client.cleanup())


if __name__ == "__main__":
    main()