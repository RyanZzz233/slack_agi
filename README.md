# MCP Invoice Management System

## Overview

This project demonstrates an AI-powered assistant that transforms traditional document management by:
- Generating invoices through intelligent, iterative data retrieval
- Providing powerful full-text search across document content
- Enabling team collaboration through reviews and comments
- Offering all functionality through a natural language interface via Slack

Traditional systems typically require rigid interfaces and manual workflows. Our approach allows users to express requests in plain language while the system handles the complexity of multi-step processes via LLM and MCP.

## Project Structure

```
HQ_MCP_v5/
├── data/                       # Data files and mock API server
│   └── data.json               # Mock order/customer data
├── mcenv/                      # Virtual environment
├── output/                     # Generated PDF files
├── server/                     # MCP server implementations
│   ├── data_server.py          # Server for retrieving order data
│   ├── pdf_server.py           # Server for HTML to PDF conversion
│   └── review_server.py        # Server for managing reviews and invoices
├── .env                        # Environment variables
├── README.md                   # This documentation
├── requirements.txt            # Project dependencies
├── reviews.db                  # SQLite database for reviews/invoices
├── slack_bot.db                # SQLite database for Slack integration
└── slack10_re.py               # Main Slack bot implementation
```

## Setup

### Prerequisites
- Python 3.10 or higher
- Slack bot token
- Anthropic API key

### Installation
1. Start the FastAPI server for mock transaction data:
   ```bash
   python data/fastapi_server.py
   ```

2. Create a virtual environment:
   ```bash
   python -m venv mcenv
   mcenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create an `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   SLACK_BOT_TOKEN=your_slack_bot_token
   ```

### Running the System
1. Start the Slack bot:
   ```bash
   python slack10_re.py
   ```

2. Set up your Slack app to forward events to your bot:
   ```
   https://your-domain.ngrok.io/slack/events
   ```

## Example Use Cases

### 1. Invoice Generation
```
User: "fetch order id 301, then use it to get the customer address info, and then form an invoice"
```

The system will:
- Retrieve order and customer details through multiple API calls
- Generate an HTML invoice
- Convert it to PDF
- Store both formats in the database
- Upload the PDF to Slack

### 2. Document Search
```
User: "search for invoice that contains order id 301"
```

The system will:
- Search across all stored invoices using full-text search
- Find matching documents
- Generate a fresh PDF
- Upload it to Slack

### 3. Collaborative Review
```
User: "create a review for the invoice I just created titled 'Q1 Finance Review'"
User: "assign this review to @john and @sarah"
User: "add comment to review: 'Please verify the tax calculation'"
```

The system will:
- Create a review record
- Assign it to specified users with notifications
- Add comments and notify participants

### 4. Task Management
```
User: "check-inbox"
User: "show me review #5"
User: "mark review #5 as completed"
```

The system will track tasks and update statuses for all users.

## How It Works

### System Components

1. **Slack Bot Interface**
   - Provides natural language user interface
   - Coordinates message routing and responses
   - Handles file uploads and notifications

2. **Claude AI**
   - Processes natural language requests
   - Determines required tools and execution sequence
   - Orchestrates multi-step workflows

3. **MCP Framework**
   - Enables tool discovery across distributed servers
   - Manages client-server communication
   - Standardizes tool registration and invocation

4. **Specialized Servers**
   - **Data Server**: Retrieves customer and order information
   - **PDF Server**: Converts HTML to PDF documents
   - **Review Server**: Manages document workflow and storage

### Data Flow Architecture

1. **Request Processing**
   - User submits natural language request via Slack
   - Request routed to Claude AI with available tool definitions
   - Claude determines required tools and execution sequence
   - System executes tools and returns results to Claude
   - Final response delivered to user with any generated files

2. **Document Generation**
   - System retrieves order and customer data via API calls
   - Data formatted into structured HTML invoice
   - HTML converted to PDF via dedicated conversion service
   - Both formats stored in database with metadata
   - PDF delivered to user via Slack

3. **Document Management**
   - Full-text search across all stored documents
   - Review workflow with assignments and inboxes
   - Comment tracking and status management
   - Automatic PDF regeneration when documents are accessed

### Technical Implementation

#### MCP Integration
- Tools registered via `@mcp.tool()` decorator with typed parameters
- Stdio-based transport layer for server communication
- JSON serialization for data exchange
- Dynamic tool discovery and unified tool registry

#### Database Architecture
- SQLite with FTS5 virtual tables for full-text search
- Relational schema for users, reviews, assignments, and documents
- Triggers maintain search index synchronization
- Transaction support for workflow operations

#### State Management
- Conversation context maintained by channel and thread
- User identity tracking: just-in-time registration
- Tool execution results preserved for multi-step operations
- Automatic context expiration for inactive conversations

#### Integration Points
- Slack API for messaging and file sharing
- Anthropic API for Claude AI capabilities
- HTTP APIs for external data retrieval
- File system for PDF storage and management

This architecture enables complex document workflows through simple natural language requests, with the system handling the technical complexity of multi-step processes, data retrieval, document generation, and collaborative review.

## Future Directions

Potential extensions include:
- Additional document types beyond invoices
- Multi-step approval workflows
- Integration with other business systems
- Data visualization capabilities
- Advanced analytics across documents

## Acknowledgments

- Anthropic for Claude AI capabilities
- The MCP protocol team
- Contributors to the open-source libraries used in this project