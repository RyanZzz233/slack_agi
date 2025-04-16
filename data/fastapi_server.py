from fastapi import FastAPI, HTTPException, Query
import json
from datetime import datetime
import os

app = FastAPI()

# Load mock data from JSON file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(os.path.dirname(current_dir), "data", "data.json")
with open(data_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Get all orders for a customer within a date range
@app.get("/orders/by-customer/{customer_id}")
def get_orders_by_customer(
    customer_id: int, 
    customer_email: str = Query(...),
    start_date: str = Query(...), 
    end_date: str = Query(...)
):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Filter orders by customer ID and date range
    orders = [
        {"order_id": o["order_id"], "order_date": o["order_date"], "total_price": o["total_price"]}
        for o in data["orders"]
        if o["customer_id"] == customer_id and 
           start_dt <= datetime.strptime(o["order_date"], "%Y-%m-%d") <= end_dt
    ]

    if not orders:
        raise HTTPException(status_code=404, detail="No orders found for the given criteria")

    return {"customer_id": customer_id, "orders": orders}

# Get full order details (items + tax & discount)
@app.get("/orders/{order_id}")
def get_order(order_id: int):
    order = next((o for o in data["orders"] if o["order_id"] == order_id), None)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    items = [item for item in data["order_items"] if item["order_id"] == order_id]
    tax_info = next((t for t in data["tax_and_discount"] if t["order_id"] == order_id), None)

    return {
        "order_id": order_id,
        "customer_id": order["customer_id"],
        "order_date": order["order_date"],
        "total_price": order["total_price"],
        "items": items,
        "tax_and_discount": tax_info or {"discount": 0, "tax": 0}
    }

# Get customer's address by customer ID
@app.get("/customers/{customer_id}/address")
def get_customer_address(customer_id: int):
    customer = next((c for c in data["customers"] if c["customer_id"] == customer_id), None)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    address = next((a for a in data["addresses"] if a["address_id"] == customer["address_id"]), None)
    if not address:
        raise HTTPException(status_code=404, detail="Address not found")

    return {
        "customer_id": customer_id,
        "customer_email": customer["email"],
        "address": {
            "street": address["street"],
            "city": address["city"],
            "zip_code": address["zip_code"]
        }
    } 