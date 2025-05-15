# MCP Invoice Management System with Planner-Executor Architecture

## Overview

This project demonstrates an AI-powered assistant that transforms traditional document management by:
- Generating invoices through intelligent, iterative data retrieval
- Providing powerful full-text search across document content
- Enabling team collaboration through reviews and comments
- Offering all functionality through a natural language interface via Slack
- **New**: Implementing a planner-executor architecture for more complex reasoning

Traditional systems typically require rigid interfaces and manual workflows. Our approach allows users to express requests in plain language while the system handles the complexity of multi-step processes via LLM and MCP.

The latest version (v11) introduces a sophisticated planner-executor architecture that mimics human cognitive processes by first analyzing problems and creating detailed execution plans before taking action.

## Project Structure

```
HQ_MCP_v11/
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
└── slack11_twin.py             # Main Slack bot implementation with planner-executor architecture
```

## Setup

### Prerequisites
- Python 3.10 or higher
- Slack bot token
- Anthropic API key
- ngrok (for exposing local server to the internet)

### Installation
1. Start the FastAPI server for mock transaction data:
   ```bash
   python data/fastapi_server.py
   ```

2. Create a virtual environment:
   ```bash
   python -m venv mcenv
   mcenv\Scripts\activate  # Windows
   source mcenv/bin/activate  # Unix/MacOS
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

### Configuring Slack App

1. Create a new Slack App at [api.slack.com/apps](https://api.slack.com/apps)

2. Under "OAuth & Permissions", add the following bot token scopes:
   - `app_mentions:read`
   - `channels:history`
   - `channels:read`
   - `chat:write`
   - `files:read`
   - `files:write`
   - `im:history`
   - `im:read`
   - `im:write`
   - `users:read`

3. Install the app to your workspace and copy the Bot User OAuth Token to your `.env` file.

4. Under "Event Subscriptions", you'll need to enable events and set up a request URL (see next section).

### Running the System with ngrok

1. Start the data server:
   ```bash
   cd data
   uvicorn fastapi_server:app --reload
   ```

2. Start the Slack bot (runs on port 3000):
   ```bash
   python slack11_twin.py
   ```

3. Download and set up ngrok from [ngrok.com](https://ngrok.com/) if you don't have it already.

4. Expose the local port using ngrok:
   ```bash
   ngrok http 3000
   ```

5. Copy the ngrok HTTPS URL (e.g., `https://a1b2c3d4.ngrok.io`) and add `/slack/events` to it:
   ```
   https://a1b2c3d4.ngrok.io/slack/events
   ```

6. Configure your Slack app's Event Subscriptions with this URL:
   - Go to your Slack app settings at [api.slack.com/apps](https://api.slack.com/apps)
   - Select "Event Subscriptions" from the sidebar
   - Toggle "Enable Events" to On
   - Enter your ngrok URL with `/slack/events` path in the Request URL field
   - Wait for Slack to verify the endpoint
   - Under "Subscribe to bot events", add `message.channels` and `message.im`
   - Save your changes

7. Reinstall your app to the workspace if required.

The ngrok URL will change each time you restart ngrok, so you'll need to update the Slack Event Subscriptions URL accordingly. For production use, you would deploy the application to a stable server with a fixed URL.

## Example Use Cases

### 1. Invoice Generation with Planning
```
User: "fetch order id 301, then use it to get the customer address info, and then form an invoice"

System: 
My thinking process:
1. Goal: Generate an invoice for order #301 with customer information
2. Required Information: Order details, customer address
3. Tools needed: Order API, customer database, invoice generator, PDF converter
4. Sequence: Get order → Get customer details → Generate invoice → Convert to PDF

**Plan for: fetch order id 301, then use it to get the customer address info, and then form an invoice**
1. Retrieve order #301 details from the system
2. Get customer information using customer ID from the order
3. Generate HTML invoice combining order and customer data
4. Convert HTML invoice to PDF
5. Save invoice to database and upload to Slack

I've created a plan to handle your request. Reply with 'execute plan' to proceed, or ask me to modify the plan.

User: "execute plan"

System:
I'll help generate an invoice for order #301.
1. Retrieved order #301: Customer ID: 103, Total: $1,256.50
2. Retrieved customer: John Smith, 123 Main St, Springfield, IL
3. Generated HTML invoice with order details and customer information
4. Converted invoice to PDF format
5. Invoice #INV-20250515-301 saved to database.
   Successfully uploaded PDF file: Invoice_301_John_Smith.pdf
```

The system will:
- Break down the request into logical subtasks
- Create a dependency graph of actions required
- Show its reasoning process to the user
- Execute the tasks in the correct sequence upon confirmation
- Provide progress updates throughout the process

### 2. Document Search
```
User: "search for invoice that contains order id 301"
```

The system will plan and execute:
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

The system will create a plan for each request and then execute it.

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

2. **Planner-Executor Architecture (New in v11)**
   - **PlannerAgent**: Analyzes requests and breaks them down into subtasks
   - **ExecutorAgent**: Handles execution of individual tasks
   - **Task Management System**: Tracks task status, dependencies, and execution order

3. **Claude AI**
   - Processes natural language requests
   - Determines required tools and execution sequence
   - Orchestrates multi-step workflows

4. **MCP Framework**
   - Enables tool discovery across distributed servers
   - Manages client-server communication
   - Standardizes tool registration and invocation

5. **Specialized Servers**
   - **Data Server**: Retrieves customer and order information
   - **PDF Server**: Converts HTML to PDF documents
   - **Review Server**: Manages document workflow and storage


### Technical Implementation

#### Task Management System (New in v11)
- `Task` class represents individual tasks with properties:
  - Status tracking (planned, in_progress, completed, failed, waiting_for_user)
  - Dependency management between tasks
  - Error handling and retry mechanism
  - Result storage for dependency resolution
- `TaskQueue` class manages collections of tasks:
  - Determines next executable task based on dependencies
  - Tracks all tasks in a workflow
  - Provides status reporting

#### Planner-Executor Architecture (New in v11)
- `PlannerAgent` analyzes user requests:
  - Breaks down complex requests into manageable subtasks
  - Identifies dependencies between tasks
  - Creates a structured plan with clear execution steps
  - Provides explanations of its thinking process
- `ExecutorAgent` handles task execution:
  - Processes tasks in the correct order
  - Manages tool selection and invocation
  - Handles failures and retries
  - Collects and formats results for the user

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
- Enhanced planning capabilities with reinforcement learning
- User feedback incorporation into planning process
- Autonomous error recovery strategies
