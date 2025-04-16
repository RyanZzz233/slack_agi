# server/pdf_converter.py
import sys
from typing import Any, Dict
import os

venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mcenv', 'Lib', 'site-packages')
sys.path.append(venv_path)

import tempfile
import time
from xhtml2pdf import pisa
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("pdf_converter")

# Create 'output' directory in the main project folder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"PDF Server initialized. Output directory: {OUTPUT_DIR}")

def generate_filename_from_html(html_content: str, prefix: str = "pdf") -> str:
    """Generate a filename based on the first <h1> tag found in the HTML content."""
    # Find the first <h1> tag in the HTML content
    start_header = html_content.find("<h1>")
    end_header = html_content.find("</h1>", start_header)
    
    if start_header != -1 and end_header != -1:
        title = html_content[start_header + 4:end_header].strip()
    else:
        title = "untitled"
    
    # Create a safe filename
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    
    # Ensure the filename is unique by appending a timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{safe_title}_{timestamp}.pdf"

def html_to_pdf(html_content: str, output_path: str) -> bool:
    """Convert HTML to PDF using xhtml2pdf"""
    with open(output_path, "wb") as output_file:
        success = pisa.CreatePDF(html_content, dest=output_file)
    return success.err == 0

@mcp.tool()
async def convert_html_to_pdf(html_content: str, title: str = "Generated PDF") -> Dict[str, Any]:
    """Convert HTML content to a PDF document.
    
    Args:
        html_content: The HTML content to convert to PDF
        title: Optional title for the PDF file
        
    Returns:
        Dictionary containing filepath of the generated PDF and status information
    """
    try:
        # Generate a safe filename
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        filename = generate_filename_from_html(html_content, safe_title)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Convert HTML to PDF
        success = html_to_pdf(html_content, filepath)
        
        if not success:
            return {
                "status": "error",
                "message": "Failed to convert HTML to PDF"
            }
            
        return {
            "status": "success",
            "message": "Successfully converted HTML to PDF",
            "filepath": filepath,
            "filename": filename
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error converting HTML to PDF: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")