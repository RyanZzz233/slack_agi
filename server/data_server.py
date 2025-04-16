from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("data_fetcher")

async def make_request(url: str, params: dict = None) -> dict[str, Any] | None:
    """Make a request with proper error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

@mcp.tool()
async def get_customer_orders(customer_id: int, customer_email: str, start_date: str, end_date: str) -> dict:
    """Get all orders for a customer within a date range"""
    url = f"http://localhost:8000/orders/by-customer/{customer_id}"
    params = {
        "customer_email": customer_email,
        "start_date": start_date,
        "end_date": end_date
    }
    
    data = await make_request(url, params)
    if not data:
        return {"error": "No orders found for the given criteria"}
    return data

@mcp.tool()
async def get_order_details(order_id: int) -> dict:
    """Get full order details including items, tax, and discount information"""
    url = f"http://localhost:8000/orders/{order_id}"
    
    data = await make_request(url)
    if not data:
        return {"error": "Order not found"}
    return data

@mcp.tool()
async def get_customer_address(customer_id: int) -> dict:
    """Get customer's address information"""
    url = f"http://localhost:8000/customers/{customer_id}/address"
    
    data = await make_request(url)
    if not data:
        return {"error": "Customer or address not found"}
    return data

if __name__ == "__main__":
    mcp.run(transport='stdio') 