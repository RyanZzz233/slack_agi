#slack11_twin.py
import asyncio
import sys
import os
import json
import threading
import time
import traceback
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from contextlib import AsyncExitStack
import glob
import re
from enum import Enum

from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class TaskStatus(Enum):
    """Enum for tracking task status"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_USER = "waiting_for_user"


class Task:
    """Representation of a task in the system"""
    
    def __init__(self, task_id: str, description: str, parent_id: str = None):
        self.task_id = task_id
        self.description = description
        self.status = TaskStatus.PLANNED
        self.dependencies: Set[str] = set()  # Task IDs this task depends on
        self.result = None  # Output of the task
        self.error = None  # Error information if task failed
        self.retry_count = 0  # Number of times this task has been retried
        self.max_retries = 3  # Maximum number of retries before giving up
        self.parent_id = parent_id  # ID of the parent task (if this is a subtask)
        self.tool_calls = []  # Record of tool calls made during execution
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary for storage"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "parent_id": self.parent_id,
            "tool_calls": self.tool_calls,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create task from dictionary"""
        task = cls(data["task_id"], data["description"], data.get("parent_id"))
        task.status = TaskStatus(data["status"])
        task.dependencies = set(data["dependencies"])
        task.result = data["result"]
        task.error = data["error"]
        task.retry_count = data["retry_count"]
        task.max_retries = data["max_retries"]
        task.tool_calls = data["tool_calls"]
        task.created_at = data["created_at"]
        task.updated_at = data["updated_at"]
        return task
    
    def add_dependency(self, task_id: str) -> None:
        """Add a dependency to this task"""
        self.dependencies.add(task_id)
    
    def mark_in_progress(self) -> None:
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = time.time()
    
    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.updated_at = time.time()
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.updated_at = time.time()
    
    def mark_waiting_for_user(self) -> None:
        """Mark task as waiting for user input"""
        self.status = TaskStatus.WAITING_FOR_USER
        self.updated_at = time.time()
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count and reset status to planned"""
        self.retry_count += 1
        self.status = TaskStatus.PLANNED
        self.updated_at = time.time()
    
    def add_tool_call(self, tool_name: str, tool_args: Dict, result: Dict) -> None:
        """Add a record of a tool call made during execution"""
        self.tool_calls.append({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "timestamp": time.time()
        })


class TaskQueue:
    """Queue for managing tasks and their dependencies"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}  # Dict of task_id -> Task
        
    def add_task(self, task: Task) -> None:
        """Add a task to the queue"""
        self.tasks[task.task_id] = task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_next_executable_task(self) -> Optional[Task]:
        """Get the next task that can be executed (all dependencies satisfied)"""
        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PLANNED:
                continue
                
            # Check if all dependencies are completed
            all_deps_completed = True
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    all_deps_completed = False
                    break
            
            if all_deps_completed:
                return task
        
        return None
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks"""
        return list(self.tasks.values())
    
    def get_incomplete_tasks(self) -> List[Task]:
        """Get all tasks that are not completed or failed"""
        return [task for task in self.tasks.values() 
                if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]]
    
    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]
    
    def get_retryable_tasks(self) -> List[Task]:
        """Get all tasks that can be retried"""
        return [task for task in self.tasks.values() if task.can_retry()]
    
    def clear(self) -> None:
        """Clear all tasks"""
        self.tasks.clear()
    
    def to_dict(self) -> Dict:
        """Convert queue to dictionary for storage"""
        return {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskQueue':
        """Create queue from dictionary"""
        queue = cls()
        for task_id, task_data in data["tasks"].items():
            queue.add_task(Task.from_dict(task_data))
        return queue


class ConversationMemory:
    """A class to store conversation history and context for each Slack channel/thread"""
    
    def __init__(self):
        self.conversations = {}  # Dict to store conversations by channel_id or thread_id
        self.context = {}  # Dict to store additional context by channel_id or thread_id
        self.session_context = {}  # Store persistent context for each session
        self.task_queues = {}  # Store task queues by session_id
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
    
    def get_task_queue(self, session_id: str) -> TaskQueue:
        """Get the task queue for a session"""
        if session_id not in self.task_queues:
            self.task_queues[session_id] = TaskQueue()
        return self.task_queues[session_id]
    
    def set_task_queue(self, session_id: str, queue: TaskQueue) -> None:
        """Set the task queue for a session"""
        self.task_queues[session_id] = queue
    
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
            if session_id in self.task_queues:
                del self.task_queues[session_id]


class PlannerAgent:
    """Agent responsible for breaking down complex tasks into subtasks and creating plans"""
    
    def __init__(self, anthropic_client: Anthropic, memory: ConversationMemory):
        self.anthropic = anthropic_client
        self.memory = memory
    
    async def create_plan(self, query: str, channel_id: str, available_tools: List[Dict]) -> Tuple[TaskQueue, str]:
        """Create a plan from a user query and return the thinking process as well"""
        # Get the current task queue or create a new one
        task_queue = self.memory.get_task_queue(channel_id)
        
        # Clear any existing tasks for a new plan
        task_queue.clear()
        
        # Structure for plan generation - main task
        main_task_id = str(uuid.uuid4())
        main_task = Task(main_task_id, f"Complete user request: {query}")
        task_queue.add_task(main_task)
        
        # Get conversation history to provide context
        conversation = self.memory.get_conversation(channel_id)
        
        # Create a copy for the API call
        messages = conversation.copy()
        
        # Add the current query with special planning instructions
        planning_query = f"""
        I need you to act as a planner. Analyze this request: "{query}"
        
        First, think step by step about this request:
        1. What is the ultimate goal the user wants to achieve?
        2. What information or data do we need to gather?
        3. What tools or actions would be required to fulfill this request?
        4. Are there any potential challenges or dependencies?
        5. What is a logical sequence of operations to achieve this goal?
        
        After your analysis, create a detailed plan with specific subtasks. For each subtask:
        1. Provide a clear description of what needs to be done
        2. Specify any dependencies (which subtasks must be completed first)
        3. Indicate which tools might be needed
        
        Format your response in two parts:
        
        PART 1: THINKING PROCESS
        Share your step-by-step reasoning about how to approach this task.
        
        PART 2: DETAILED PLAN
        Format your plan as a JSON array of task objects, like this:
        ```json
        [
          {{
            "task_id": "1",
            "description": "Description of first task",
            "dependencies": [],
            "tools": ["tool_name1", "tool_name2"]
          }},
          {{
            "task_id": "2",
            "description": "Description of second task",
            "dependencies": ["1"],
            "tools": ["tool_name3"]
          }}
        ]
        ```
        """
        
        messages.append({
            "role": "user",
            "content": planning_query
        })
        
        # Create enhanced system prompt for planner
        base_planner_prompt = """You are an AI planning assistant that breaks down complex tasks into logical, actionable subtasks.
        
        When given a user request, your job is to:
        1. Analyze the user's intention and end goal
        2. Break down the goal into sequential subtasks
        3. Identify dependencies between subtasks
        4. Determine which tools would be needed for each subtask
        
        Consider both explicit and implicit steps needed. For example, if the user asks to "create an invoice and send it for review", 
        you should include steps for gathering data, creating the invoice, saving it to the database, 
        creating a review, assigning it to someone, and notifying them.
        
        Make your thinking process transparent and provide a detailed plan as requested.
        """
        
        system_prompt = self.memory.get_enhanced_system_prompt(channel_id, base_planner_prompt)
        
        # Make Claude API call to generate the plan
        try:
            # Important: No await here - this was causing the error
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                system=system_prompt,
                messages=messages
            )
            
            # Extract the thinking process and JSON plan from response
            response_text = response.content[0].text
            
            # Extract thinking process
            thinking_process = ""
            if "PART 1: THINKING PROCESS" in response_text:
                thinking_parts = response_text.split("PART 1: THINKING PROCESS", 1)
                if len(thinking_parts) > 1:
                    second_part = thinking_parts[1]
                    if "PART 2: DETAILED PLAN" in second_part:
                        thinking_process = second_part.split("PART 2: DETAILED PLAN", 1)[0].strip()
                    else:
                        thinking_process = second_part.strip()
            
            # Extract JSON plan - look for patterns that might indicate the JSON
            plan_text = ""
            if "```json" in response_text:
                plan_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_blocks = re.findall(r'```(.*?)```', response_text, re.DOTALL)
                # Try to find a JSON block
                for block in json_blocks:
                    try:
                        # Strip any language identifier if present
                        if block.startswith('json\n'):
                            block = block[5:]
                        json.loads(block)
                        plan_text = block
                        break
                    except:
                        continue
            
            # If no JSON found with delimiters, look for anything that looks like JSON array
            if not plan_text:
                matches = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                if matches:
                    plan_text = matches.group(0)
            
            # Try to parse the plan_text as JSON
            try:
                plan_data = json.loads(plan_text)
            except json.JSONDecodeError:
                # If parsing fails, create a simple default plan
                print(f"Error parsing plan JSON: {traceback.format_exc()}")
                plan_data = [{"task_id": "1", "description": query, "dependencies": [], "tools": []}]
            
            # Create tasks from the plan
            task_id_map = {}  # Map from plan task_ids to our UUIDs
            
            for task_data in plan_data:
                plan_task_id = str(task_data["task_id"])
                new_task_id = str(uuid.uuid4())
                task_id_map[plan_task_id] = new_task_id
                
                task = Task(new_task_id, task_data["description"], parent_id=main_task_id)
                task_queue.add_task(task)
            
            # Now add dependencies using our UUID mapping
            for task_data in plan_data:
                plan_task_id = str(task_data["task_id"])
                our_task_id = task_id_map[plan_task_id]
                task = task_queue.get_task(our_task_id)
                
                if task:
                    for dep_id in task_data.get("dependencies", []):
                        dep_id = str(dep_id)
                        if dep_id in task_id_map:
                            task.add_dependency(task_id_map[dep_id])
                    
                    # Make the main task depend on all subtasks
                    main_task.add_dependency(our_task_id)
            
            # Store the plan in memory
            self.memory.set_task_queue(channel_id, task_queue)
            
            # Store plan explanation in context for reference
            self.memory.store_context(channel_id, "current_plan", plan_data)
            self.memory.store_context(channel_id, "plan_thinking", thinking_process)
            
            return task_queue, thinking_process
            
        except Exception as e:
            print(f"Error creating plan: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            
            # Create a minimal task queue with an error message
            error_task = Task(str(uuid.uuid4()), f"Error: {str(e)}", parent_id=main_task_id)
            task_queue.add_task(error_task)
            main_task.add_dependency(error_task.task_id)
            
            return task_queue, f"Error creating plan: {str(e)}"
    
    def format_plan_for_user(self, task_queue: TaskQueue, query: str) -> str:
        """Format the plan for presentation to the user"""
        tasks = [t for t in task_queue.get_all_tasks() if t.parent_id is not None]
        
        # Sort tasks by dependencies
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        # Start with tasks with no dependencies
        no_deps = [t for t in remaining_tasks if not t.dependencies]
        sorted_tasks.extend(no_deps)
        for t in no_deps:
            if t in remaining_tasks:
                remaining_tasks.remove(t)
        
        # Iteratively add tasks whose dependencies are satisfied
        while remaining_tasks:
            added_this_round = False
            for task in remaining_tasks[:]:
                deps_satisfied = all(dep in [t.task_id for t in sorted_tasks] for dep in task.dependencies)
                if deps_satisfied:
                    sorted_tasks.append(task)
                    remaining_tasks.remove(task)
                    added_this_round = True
            
            # If we didn't add any tasks in this iteration, there must be a dependency cycle
            # Just add the remaining tasks to avoid an infinite loop
            if not added_this_round and remaining_tasks:
                sorted_tasks.extend(remaining_tasks)
                break
        
        # Generate a human-readable plan
        plan_text = f"**Plan for: {query}**\n\n"
        for i, task in enumerate(sorted_tasks, 1):
            plan_text += f"{i}. {task.description}\n"
            
        return plan_text


class ExecutorAgent:
    """Agent responsible for executing tasks and handling errors"""
    
    def __init__(self, anthropic_client: Anthropic, memory: ConversationMemory, mcp_client: 'MCPClient'):
        self.anthropic = anthropic_client
        self.memory = memory
        self.mcp_client = mcp_client
        self.results_buffer = []  # Store results to be returned to the user
    
    async def execute_plan(self, channel_id: str, user_id: str = None, username: str = None) -> str:
        """Execute the current plan for a channel"""
        # Get the task queue
        task_queue = self.memory.get_task_queue(channel_id)
        
        # Clear the results buffer
        self.results_buffer = []
        
        # Find all tasks that are planned
        planned_tasks = [t for t in task_queue.get_all_tasks() if t.status == TaskStatus.PLANNED]
        if not planned_tasks:
            return "No tasks to execute. Please create a plan first."
        
        # Get a count of total tasks (excluding the main task)
        total_tasks = len([t for t in task_queue.get_all_tasks() if t.parent_id is not None])
        completed_tasks = 0
        
        # Process the next executable task until there are none left or we hit an error
        max_iterations = 10  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get the next executable task
            next_task = task_queue.get_next_executable_task()
            
            if not next_task:
                # Check if all tasks are complete
                incomplete_tasks = task_queue.get_incomplete_tasks()
                if not incomplete_tasks:
                    # All tasks complete!
                    main_task = None
                    for task in task_queue.get_all_tasks():
                        if not task.parent_id:  # This is the main task
                            main_task = task
                            break
                    
                    if main_task:
                        main_task.mark_completed()
                    
                    # Return the complete buffer of results
                    return "\n\n".join(self.results_buffer)
                
                # Check for failed tasks that can be retried
                retryable_tasks = task_queue.get_retryable_tasks()
                if retryable_tasks:
                    task = retryable_tasks[0]
                    task.increment_retry()
                    continue
                
                # If there are no retryable tasks but we still have incomplete tasks,
                # there must be a dependency issue or all tasks have failed
                failed_msg = "Some tasks failed and cannot be retried."
                self.results_buffer.append(failed_msg)
                
                # Detail the failed tasks
                failed_tasks = task_queue.get_failed_tasks()
                for task in failed_tasks:
                    self.results_buffer.append(f"Failed: {task.description} - Error: {task.error}")
                
                return "\n\n".join(self.results_buffer)
            
            # We have a task to execute
            next_task.mark_in_progress()
            
            # Execute the task
            result = await self.execute_task(next_task, channel_id, user_id, username)
            
            if result["success"]:
                next_task.mark_completed(result["result"])
                completed_tasks += 1
                
                # Instead of adding progress messages to the buffer, just add any content from the result
                if "content" in result:
                    self.results_buffer.append(result["content"])
            else:
                next_task.mark_failed(result["error"])
                
                # Check if we can retry
                if not next_task.can_retry():
                    self.results_buffer.append(f"Failed to complete task: {result['error']}")
        
        # If we hit the iteration limit
        self.results_buffer.append("Reached maximum execution steps without completing all tasks.")
        return "\n\n".join(self.results_buffer)
    
    async def execute_task(self, task: Task, channel_id: str, user_id: str = None, username: str = None) -> Dict:
        """Execute a single task"""
        try:
            # Format the task description for Claude
            task_description = f"Execute this task: {task.description}"
            
            # Get conversation history for context
            conversation = self.memory.get_conversation(channel_id)
            
            # Create a copy for the API call
            messages = conversation.copy()
            
            # Get enhanced system prompt for executor
            base_executor_prompt = """You are an AI executor agent that performs specific tasks to accomplish a larger goal.
            
            You will be given a specific task to complete. Your job is to:
            1. Understand exactly what needs to be done
            2. Select the appropriate tools to complete the task
            3. Execute the task step by step
            4. Handle any errors or unexpected situations
            5. Report the result clearly and directly

            Your responses should be concise and focused on delivering the requested information or completing the requested action.
            Do not include progress indicators, emojis, or unnecessary explanations.
            
            When you complete the task, respond with just the information or confirmation that would be valuable to the user.
            If you encounter an error you can't resolve, respond with a clear explanation of the error.
            """
            
            system_prompt = self.memory.get_enhanced_system_prompt(channel_id, base_executor_prompt)
            
            # Add task context - previous task results that this task depends on
            task_context = "Here is context from previous tasks that might be relevant:\n"
            context_added = False
            
            for dep_id in task.dependencies:
                dep_task = self.memory.get_task_queue(channel_id).get_task(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED and dep_task.result:
                    task_context += f"\nResult from task '{dep_task.description}': {json.dumps(dep_task.result)}\n"
                    context_added = True
            
            if context_added:
                task_description = f"{task_context}\n\n{task_description}"
            
            # Add the task to execute
            messages.append({
                "role": "user",
                "content": task_description
            })
            
            # Collect tools from all connected servers
            available_tools = []
            for server_path, server_data in self.mcp_client.sessions.items():
                # Create a clean server ID by replacing invalid characters
                server_id = server_path.replace('/', '_').replace('.', '_').replace('\\', '_')
                response = await server_data['session'].list_tools()
                for tool in response.tools:
                    available_tools.append({
                        "name": f"{server_id}__{tool.name}",  # Use double underscore as separator
                        "description": f"[{server_path}] {tool.description}",  # Add server path to description
                        "input_schema": tool.inputSchema
                    })
            
            # Make Claude API call to execute the task - NO AWAIT HERE
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=messages,
                    tools=available_tools
                )
                
                # Process the response
                requires_tool_call = False
                tool_results = []
                all_results = []
                response_text = ""
                task_response = ""  # This will be the content returned to the user
                
                # Storage for the assistant's response parts
                assistant_message_content = []
                
                # Process each part of the response
                for content in response.content:
                    if content.type == 'text':
                        response_text += content.text
                        task_response += content.text  # Add to result that will be shown to user
                        assistant_message_content.append(content)
                        
                    elif content.type == 'tool_use':
                        requires_tool_call = True
                        # Parse server ID and tool name
                        server_id, tool_name = content.name.split('__', 1)
                        
                        # Find the original server path
                        server_path = next(
                            path for path in self.mcp_client.sessions.keys() 
                            if server_id == path.replace('/', '_').replace('.', '_').replace('\\', '_')
                        )
                        tool_args = content.input
                        
                        # If tool needs user_id and it's not provided, add it from context
                        if user_id and "user_id" in tool_args and not tool_args.get("user_id"):
                            tool_args["user_id"] = user_id
                            
                        # Execute tool call on the appropriate server
                        result = await self.mcp_client.sessions[server_path]['session'].call_tool(tool_name, tool_args)
                        
                        # Parse result
                        result_data = self.mcp_client.parse_tool_result(result.content)
                        
                        # Record the tool call in the task
                        task.add_tool_call(tool_name, tool_args, result_data)
                        
                        # Add tool result to the task_response - this is important for the user to see
                        if "success" in result_data and result_data["success"]:
                            # Format different types of successful responses for the user to see
                            if tool_name == "get_inbox":
                                if "reviews" in result_data and result_data["reviews"]:
                                    task_response += "\n\nYour pending reviews:\n"
                                    for i, review in enumerate(result_data["reviews"], 1):
                                        task_response += f"{i}. Review #{review.get('id', 'Unknown')}: {review.get('title', 'Untitled')}\n"
                                else:
                                    task_response += "\n\nYou don't have any pending reviews."
                            elif tool_name == "get_review_details":
                                if "review" in result_data:
                                    review = result_data["review"]
                                    task_response += f"\n\nReview #{review.get('id', 'Unknown')}: {review.get('title', 'Untitled')}\n"
                                    task_response += f"Status: {review.get('status', 'Unknown')}\n"
                                    task_response += f"Description: {review.get('description', 'No description')}\n"
                                    if "comments" in result_data and result_data["comments"]:
                                        task_response += "\nComments:\n"
                                        for i, comment in enumerate(result_data["comments"], 1):
                                            task_response += f"{i}. {comment.get('username', 'Unknown')}: {comment.get('content', '')}\n"
                            elif tool_name == "search_invoices":
                                if "results" in result_data and result_data["results"]:
                                    task_response += "\n\nFound invoices:\n"
                                    for i, invoice in enumerate(result_data["results"], 1):
                                        task_response += f"{i}. Invoice #{invoice.get('id', 'Unknown')}: {invoice.get('invoice_number', 'Unknown')} - {invoice.get('client_name', 'Unknown Client')}\n"
                                else:
                                    task_response += "\n\nNo invoices found matching your search."
                        
                        # Process specific post-tool actions
                        if tool_name in ["json_to_pdf", "convert_html_to_pdf"]:
                            success, message = await self.mcp_client.handle_pdf_generation_result(
                                channel_id=channel_id,
                                result=result.content,
                                user_id=user_id
                            )
                            if success:
                                task_response += f"\n\n{message}"
                            else:
                                return {"success": False, "error": message}
                        
                        # Handle invoice search and generation
                        if tool_name in ["search_invoices", "get_invoice_by_id", "get_invoice_by_number"] and result_data.get("success"):
                            if tool_name == "search_invoices" and result_data.get("results") and len(result_data.get("results", [])) > 0:
                                invoice_id = result_data["results"][0]["id"]
                                success, message = await self.mcp_client.process_invoice_request(
                                    channel_id=channel_id, 
                                    invoice_id=invoice_id,
                                    user_id=user_id
                                )
                                if success:
                                    task_response += f"\n\nI've automatically generated a PDF of this invoice for you."
                                else:
                                    return {"success": False, "error": message}
                            
                            elif tool_name in ["get_invoice_by_id", "get_invoice_by_number"] and result_data.get("invoice", {}).get("html_content"):
                                invoice_id = result_data["invoice"]["id"]
                                success, message = await self.mcp_client.process_invoice_request(
                                    channel_id=channel_id, 
                                    invoice_id=invoice_id,
                                    user_id=user_id
                                )
                                if success:
                                    task_response += f"\n\nI've automatically generated a PDF of this invoice for you."
                                else:
                                    return {"success": False, "error": message}
                        
                        # Store the tool result
                        tool_results.append({
                            "tool": content.name,
                            "args": tool_args,
                            "result": result_data
                        })
                        
                        # Add the assistant's response (with tool call) to the conversation history
                        self.memory.add_message(channel_id, {
                            "role": "assistant",
                            "content": [content]
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
                
                return {
                    "success": True, 
                    "result": {"summary": "Task completed", "tools": tool_results},
                    "content": task_response.strip()  # Include the response content for the user
                }
                
            except Exception as e:
                print(f"Execution API call error: {str(e)}")
                traceback_info = traceback.format_exc()
                print(f"Traceback: {traceback_info}")
                return {"success": False, "error": f"API call error: {str(e)}"}
            
        except Exception as e:
            print(f"Error executing task: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return {"success": False, "error": f"Execution error: {str(e)}"}


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
        """Process a query using the planner-executor architecture"""
        # Initialize planner and executor if they don't exist yet
        if not hasattr(self, 'planner'):
            self.planner = PlannerAgent(self.anthropic, self.memory)
            self.executor = ExecutorAgent(self.anthropic, self.memory, self)
        
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
        
        # Check for special commands
        if query.lower().strip().startswith("execute plan"):
            # Execute the current plan
            return await self.executor.execute_plan(channel_id, user_id, username)
            
        # Regular query - create a plan
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
        
        # First, use the planner to create a plan
        task_queue, thinking_process = await self.planner.create_plan(query, channel_id, available_tools)
        
        # Format the plan for the user
        plan_text = self.planner.format_plan_for_user(task_queue, query)
        
        # Combine thinking process and plan
        response_text = f"My thinking process:\n\n{thinking_process}\n\n"
        response_text += f"{plan_text}\n\n"
        response_text += "I've created a plan to handle your request. Reply with 'execute plan' to proceed, or ask me to modify the plan."
        
        return response_text

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
    
    print("Starting MCP Slack Bot with Planner-Executor Architecture...")
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