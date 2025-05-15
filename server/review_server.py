#server/review_server.py
#!/usr/bin/env python3
"""
Review Management MCP Server with Invoice Storage and Retrieval

This server provides tools for managing document reviews, user tracking,
invoice storage/retrieval, and notifications between team members.
"""

import sys
import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any

# Handle virtual environment path if needed
venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mcenv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.append(venv_path)

# Import FastMCP
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("review_manager")

class ReviewDatabase:
    """Database manager for the review system"""
    
    def __init__(self, db_path="reviews.db"):
        self.db_path = db_path
        self.initialize_db()
    
    def initialize_db(self):
        """Create the database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create reviews table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            file_path TEXT,
            file_type TEXT,
            creator_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            slack_file_id TEXT,
            slack_permalink TEXT,
            FOREIGN KEY (creator_id) REFERENCES users (id)
        )
        ''')
        
        # Create review assignments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            assignee_id TEXT NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (review_id) REFERENCES reviews (id),
            FOREIGN KEY (assignee_id) REFERENCES users (id)
        )
        ''')
        
        # Create comments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (review_id) REFERENCES reviews (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create invoices table for storing generated invoices
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number TEXT,
            client_name TEXT,
            client_email TEXT,
            amount REAL,
            currency TEXT DEFAULT 'USD',
            description TEXT,
            html_content TEXT NOT NULL,
            pdf_path TEXT,
            creator_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (creator_id) REFERENCES users (id)
        )
        ''')
        
        # Create FTS virtual table for invoice searching
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS invoices_fts USING fts5(
            invoice_number,
            client_name,
            client_email,
            description,
            html_content,
            content='invoices',
            content_rowid='id'
        )
        ''')
        
        # Create triggers to keep FTS table in sync
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS invoices_ai AFTER INSERT ON invoices BEGIN
            INSERT INTO invoices_fts(rowid, invoice_number, client_name, client_email, description, html_content)
            VALUES (new.id, new.invoice_number, new.client_name, new.client_email, new.description, new.html_content);
        END
        ''')
        
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS invoices_ad AFTER DELETE ON invoices BEGIN
            INSERT INTO invoices_fts(invoices_fts, rowid, invoice_number, client_name, client_email, description, html_content)
            VALUES('delete', old.id, old.invoice_number, old.client_name, old.client_email, old.description, old.html_content);
        END
        ''')
        
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS invoices_au AFTER UPDATE ON invoices BEGIN
            INSERT INTO invoices_fts(invoices_fts, rowid, invoice_number, client_name, client_email, description, html_content)
            VALUES('delete', old.id, old.invoice_number, old.client_name, old.client_email, old.description, old.html_content);
            INSERT INTO invoices_fts(rowid, invoice_number, client_name, client_email, description, html_content)
            VALUES (new.id, new.invoice_number, new.client_name, new.client_email, new.description, new.html_content);
        END
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")

# Create a global instance of the database
db = ReviewDatabase()

@mcp.tool()
async def register_user(user_id: str, username: str) -> Dict[str, Any]:
    """
    Register a user in the review system or update existing user info
    
    Args:
        user_id: The Slack user ID
        username: The Slack username
    
    Returns:
        Dict: Status information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Update username if it's changed
            cursor.execute("UPDATE users SET username = ? WHERE id = ?", 
                          (username, user_id))
        else:
            # Create new user
            cursor.execute("INSERT INTO users (id, username) VALUES (?, ?)",
                          (user_id, username))
        
        conn.commit()
        conn.close()
        return {"success": True, "user_id": user_id, "username": username}
    except Exception as e:
        print(f"Error registering user: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_user(user_id: Optional[str] = None, username: Optional[str] = None) -> Dict[str, Any]:
    """
    Get user information by user_id or username
    
    Args:
        user_id: The Slack user ID (optional)
        username: The Slack username (optional)
    
    Returns:
        Dict: User information
    """
    try:
        if not user_id and not username:
            return {"success": False, "error": "Either user_id or username must be provided"}
        
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Try to find the user by both methods if either is provided
        if user_id:
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            if user:
                conn.close()
                return {
                    "success": True,
                    "user": dict(user)
                }
            
        if username:
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user:
                conn.close()
                return {
                    "success": True,
                    "user": dict(user)
                }
        
        # If we got here, no user was found
        conn.close()
        return {"success": False, "error": "User not found"}
    except Exception as e:
        print(f"Error getting user: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_my_user_info(user_id: str) -> Dict[str, Any]:
    """
    Get your own user information
    
    Args:
        user_id: The Slack user ID of the requesting user
    
    Returns:
        Dict: User information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if user:
            result = dict(user)
            conn.close()
            return {
                "success": True,
                "user": result
            }
        else:
            conn.close()
            return {
                "success": False, 
                "error": "You are not registered in the system yet. Please provide your username to register."
            }
    except Exception as e:
        print(f"Error in get_my_user_info: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def create_review(title: str, creator_id: str, description: str = "",
                 file_path: str = "", file_type: str = "") -> Dict[str, Any]:
    """
    Create a new review
    
    Args:
        title: The review title
        creator_id: The Slack ID of the creator
        description: Description of the review (optional)
        file_path: Path to the file being reviewed (optional)
        file_type: Type of file (pdf, doc, etc.) (optional)
    
    Returns:
        Dict: Review information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO reviews (title, description, file_path, file_type, creator_id)
        VALUES (?, ?, ?, ?, ?)
        """, (title, description, file_path, file_type, creator_id))
        
        review_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "review_id": review_id,
            "title": title,
            "creator_id": creator_id
        }
    except Exception as e:
        print(f"Error creating review: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def update_review_file_info(review_id: int, slack_file_id: str = None, slack_permalink: str = None) -> Dict[str, Any]:
    """
    Update file information for a review
    
    Args:
        review_id: The ID of the review
        slack_file_id: The Slack file ID (optional)
        slack_permalink: The Slack permalink to the file (optional)
    
    Returns:
        Dict: Status information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Build update query based on provided parameters
        update_parts = []
        params = []
        
        if slack_file_id:
            update_parts.append("slack_file_id = ?")
            params.append(slack_file_id)
            
        if slack_permalink:
            update_parts.append("slack_permalink = ?")
            params.append(slack_permalink)
            
        if not update_parts:
            return {"success": False, "error": "No update parameters provided"}
            
        # Add review_id to params
        params.append(review_id)
        
        # Execute update
        cursor.execute(f"""
        UPDATE reviews SET {', '.join(update_parts)}
        WHERE id = ?
        """, params)
        
        if cursor.rowcount == 0:
            conn.close()
            return {"success": False, "error": "Review not found"}
            
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "review_id": review_id,
            "message": "Review file info updated successfully"
        }
    except Exception as e:
        print(f"Error updating review file info: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def assign_review(review_id: int, assignee_id: str) -> Dict[str, Any]:
    """
    Assign a review to a user
    
    Args:
        review_id: The ID of the review
        assignee_id: The Slack ID of the assignee
    
    Returns:
        Dict: Assignment information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Check if review exists
        cursor.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        if not cursor.fetchone():
            return {"success": False, "error": "Review not found"}
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = ?", (assignee_id,))
        if not cursor.fetchone():
            return {"success": False, "error": "User not found"}
        
        # Check if already assigned
        cursor.execute("""
        SELECT id FROM review_assignments 
        WHERE review_id = ? AND assignee_id = ?
        """, (review_id, assignee_id))
        
        if cursor.fetchone():
            return {"success": False, "error": "Review already assigned to this user"}
        
        # Create assignment
        cursor.execute("""
        INSERT INTO review_assignments (review_id, assignee_id)
        VALUES (?, ?)
        """, (review_id, assignee_id))
        
        assignment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "assignment_id": assignment_id,
            "review_id": review_id,
            "assignee_id": assignee_id
        }
    except Exception as e:
        print(f"Error assigning review: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_inbox(user_id: str, status: str = "pending") -> Dict[str, Any]:
    """
    Get a user's pending reviews
    
    Args:
        user_id: The Slack ID of the user
        status: Filter by status (pending, completed, all)
    
    Returns:
        Dict: List of reviews assigned to the user
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT r.id, r.title, r.description, r.file_path, r.file_type, 
               r.created_at, r.status as review_status,
               u.username as creator_username,
               ra.id as assignment_id, ra.assigned_at, ra.completed_at, ra.status as assignment_status
        FROM reviews r
        JOIN users u ON r.creator_id = u.id
        JOIN review_assignments ra ON r.id = ra.review_id
        WHERE ra.assignee_id = ?
        """
        
        params = [user_id]
        
        if status != "all":
            query += " AND ra.status = ?"
            params.append(status)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        reviews = []
        for row in rows:
            reviews.append(dict(row))
        
        conn.close()
        
        return {
            "success": True,
            "reviews": reviews,
            "count": len(reviews)
        }
    except Exception as e:
        print(f"Error getting inbox: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def add_comment(review_id: int, user_id: str, content: str) -> Dict[str, Any]:
    """
    Add a comment to a review
    
    Args:
        review_id: The ID of the review
        user_id: The Slack ID of the commenter
        content: The comment text
    
    Returns:
        Dict: Comment information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Check if review exists
        cursor.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        if not cursor.fetchone():
            return {"success": False, "error": "Review not found"}
        
        # Add comment
        cursor.execute("""
        INSERT INTO comments (review_id, user_id, content)
        VALUES (?, ?, ?)
        """, (review_id, user_id, content))
        
        comment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "comment_id": comment_id,
            "review_id": review_id
        }
    except Exception as e:
        print(f"Error adding comment: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_comments(review_id: int) -> Dict[str, Any]:
    """
    Get all comments for a review
    
    Args:
        review_id: The ID of the review
    
    Returns:
        Dict: List of comments
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if review exists
        cursor.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        if not cursor.fetchone():
            return {"success": False, "error": "Review not found"}
        
        # Get comments
        cursor.execute("""
        SELECT c.id, c.content, c.created_at, u.username
        FROM comments c
        JOIN users u ON c.user_id = u.id
        WHERE c.review_id = ?
        ORDER BY c.created_at ASC
        """, (review_id,))
        
        rows = cursor.fetchall()
        comments = []
        for row in rows:
            comments.append(dict(row))
        
        conn.close()
        
        return {
            "success": True,
            "comments": comments,
            "count": len(comments)
        }
    except Exception as e:
        print(f"Error getting comments: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def complete_review(assignment_id: int) -> Dict[str, Any]:
    """
    Mark a review assignment as completed
    
    Args:
        assignment_id: The ID of the assignment
    
    Returns:
        Dict: Status information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Check if assignment exists
        cursor.execute("SELECT id, review_id FROM review_assignments WHERE id = ?", (assignment_id,))
        result = cursor.fetchone()
        if not result:
            return {"success": False, "error": "Assignment not found"}
        
        review_id = result[1]
        
        # Update assignment status
        cursor.execute("""
        UPDATE review_assignments 
        SET status = 'completed', completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """, (assignment_id,))
        
        # Check if all assignments for this review are completed
        cursor.execute("""
        SELECT COUNT(*) FROM review_assignments 
        WHERE review_id = ? AND status != 'completed'
        """, (review_id,))
        
        pending_count = cursor.fetchone()[0]
        
        # If all assignments are completed, update review status
        if pending_count == 0:
            cursor.execute("""
            UPDATE reviews SET status = 'completed'
            WHERE id = ?
            """, (review_id,))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "assignment_id": assignment_id,
            "status": "completed",
            "all_completed": pending_count == 0
        }
    except Exception as e:
        print(f"Error completing review: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_review_details(review_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a review
    
    Args:
        review_id: The ID of the review
    
    Returns:
        Dict: Review details including assignments and comments
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get review details
        cursor.execute("""
        SELECT r.*, u.username as creator_username
        FROM reviews r
        JOIN users u ON r.creator_id = u.id
        WHERE r.id = ?
        """, (review_id,))
        
        review = cursor.fetchone()
        if not review:
            return {"success": False, "error": "Review not found"}
        
        review_dict = dict(review)
        
        # Get assignments
        cursor.execute("""
        SELECT ra.*, u.username as assignee_username
        FROM review_assignments ra
        JOIN users u ON ra.assignee_id = u.id
        WHERE ra.review_id = ?
        """, (review_id,))
        
        assignments = [dict(row) for row in cursor.fetchall()]
        
        # Get comments
        cursor.execute("""
        SELECT c.*, u.username
        FROM comments c
        JOIN users u ON c.user_id = u.id
        WHERE c.review_id = ?
        ORDER BY c.created_at ASC
        """, (review_id,))
        
        comments = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "success": True,
            "review": review_dict,
            "assignments": assignments,
            "comments": comments
        }
    except Exception as e:
        print(f"Error getting review details: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def list_all_users() -> Dict[str, Any]:
    """
    List all registered users in the system
    
    Returns:
        Dict: List of all registered users
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users ORDER BY username")
        users = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "success": True,
            "users": users,
            "count": len(users)
        }
    except Exception as e:
        print(f"Error listing users: {e}")
        return {"success": False, "error": str(e)}

# New tools for invoice management and retrieval

@mcp.tool()
async def save_invoice(invoice_number: str, client_name: str, html_content: str, 
                creator_id: str, client_email: str = "", amount: float = 0.0,
                currency: str = "USD", description: str = "", pdf_path: str = "") -> Dict[str, Any]:
    """
    Save an invoice in the database
    
    Args:
        invoice_number: The invoice number or ID
        client_name: The client's name
        html_content: The full HTML content of the invoice
        creator_id: The Slack ID of the creator
        client_email: Client's email address (optional)
        amount: The invoice amount (optional)
        currency: The currency code (optional, defaults to USD)
        description: Description of the invoice (optional)
        pdf_path: Path to the generated PDF (optional)
    
    Returns:
        Dict: Status information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO invoices (
            invoice_number, client_name, client_email, amount, currency, 
            description, html_content, pdf_path, creator_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            invoice_number, client_name, client_email, amount, currency,
            description, html_content, pdf_path, creator_id
        ))
        
        invoice_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "invoice_id": invoice_id,
            "invoice_number": invoice_number,
            "client_name": client_name
        }
    except Exception as e:
        print(f"Error saving invoice: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def search_invoices(query: str) -> Dict[str, Any]:
    """
    Search for invoices using full-text search
    
    Args:
        query: Search terms to find matching invoices
    
    Returns:
        Dict: List of matching invoices
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use FTS5 to search
        cursor.execute("""
        SELECT 
            i.id, i.invoice_number, i.client_name, i.client_email, 
            i.amount, i.currency, i.description, i.pdf_path,
            i.created_at, u.username as creator_username,
            snippet(invoices_fts, -1, '<mark>', '</mark>', '...', 15) as snippet
        FROM invoices_fts
        JOIN invoices i ON invoices_fts.rowid = i.id
        JOIN users u ON i.creator_id = u.id
        WHERE invoices_fts MATCH ?
        ORDER BY rank
        LIMIT 10
        """, (query,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        print(f"Error searching invoices: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_invoice_by_id(invoice_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific invoice by ID
    
    Args:
        invoice_id: The ID of the invoice
    
    Returns:
        Dict: Invoice details including HTML content
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT i.*, u.username as creator_username
        FROM invoices i
        JOIN users u ON i.creator_id = u.id
        WHERE i.id = ?
        """, (invoice_id,))
        
        invoice = cursor.fetchone()
        if not invoice:
            return {"success": False, "error": "Invoice not found"}
        
        invoice_dict = dict(invoice)
        conn.close()
        
        return {
            "success": True,
            "invoice": invoice_dict
        }
    except Exception as e:
        print(f"Error retrieving invoice: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_invoice_by_number(invoice_number: str) -> Dict[str, Any]:
    """
    Retrieve a specific invoice by invoice number
    
    Args:
        invoice_number: The invoice number
    
    Returns:
        Dict: Invoice details including HTML content
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT i.*, u.username as creator_username
        FROM invoices i
        JOIN users u ON i.creator_id = u.id
        WHERE i.invoice_number = ?
        """, (invoice_number,))
        
        invoice = cursor.fetchone()
        if not invoice:
            return {"success": False, "error": "Invoice not found"}
        
        invoice_dict = dict(invoice)
        conn.close()
        
        return {
            "success": True,
            "invoice": invoice_dict
        }
    except Exception as e:
        print(f"Error retrieving invoice: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_invoices_by_client(client_name: str) -> Dict[str, Any]:
    """
    Get all invoices for a specific client
    
    Args:
        client_name: The client's name
    
    Returns:
        Dict: List of invoices for the client
    """
    try:
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT i.id, i.invoice_number, i.client_name, i.amount, i.currency, 
               i.description, i.created_at, u.username as creator_username
        FROM invoices i
        JOIN users u ON i.creator_id = u.id
        WHERE i.client_name LIKE ?
        ORDER BY i.created_at DESC
        """, (f"%{client_name}%",))
        
        rows = cursor.fetchall()
        invoices = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "success": True,
            "invoices": invoices,
            "count": len(invoices)
        }
    except Exception as e:
        print(f"Error retrieving client invoices: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def update_invoice_pdf_path(invoice_id: int, pdf_path: str) -> Dict[str, Any]:
    """
    Update the PDF path for an invoice
    
    Args:
        invoice_id: The ID of the invoice
        pdf_path: The new PDF file path
    
    Returns:
        Dict: Status information
    """
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE invoices SET pdf_path = ?
        WHERE id = ?
        """, (pdf_path, invoice_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return {"success": False, "error": "Invoice not found"}
            
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "invoice_id": invoice_id,
            "message": "Invoice PDF path updated successfully"
        }
    except Exception as e:
        print(f"Error updating invoice PDF path: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Starting Review Management MCP Server with Invoice Storage and Retrieval...")
    mcp.run(transport="stdio")