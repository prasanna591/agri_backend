from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from .marketplace import products_db, Product

router = APIRouter()

# --------------------
# Models
# --------------------
class Order(BaseModel):
    id: int
    product_ids: List[int]
    total_price: float

class OrderCreate(BaseModel):
    product_ids: List[int]

# In-memory orders store
orders_db: List[Order] = []

# --------------------
# Endpoints
# --------------------
@router.get("/all", response_model=List[Order])
def get_all_orders():
    return orders_db

@router.post("/create", response_model=Order)
def create_order(order_req: OrderCreate):
    selected_products = []
    total = 0.0

    for pid in order_req.product_ids:
        product = next((p for p in products_db if p.id == pid), None)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product ID {pid} not found")
        selected_products.append(product)
        total += product.price

    new_order = Order(
        id=len(orders_db) + 1,
        product_ids=order_req.product_ids,
        total_price=total,
    )
    orders_db.append(new_order)
    return new_order
