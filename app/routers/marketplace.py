from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

# --------------------
# Models
# --------------------
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: str = ""

# In-memory product store
products_db: List[Product] = [
    Product(id=1, name="Organic Fertilizer", price=299.0, description="Rich in nitrogen and potassium."),
    Product(id=2, name="Hybrid Seeds Pack", price=149.0, description="High-yield, drought-resistant."),
    Product(id=3, name="Drip Irrigation Kit", price=899.0, description="Water-efficient irrigation system."),
]

# --------------------
# Endpoints
# --------------------
@router.get("/all", response_model=List[Product])
def get_all_products():
    return products_db

@router.post("/add", response_model=Product)
def add_product(product: Product):
    # Prevent duplicate IDs
    if any(p.id == product.id for p in products_db):
        raise HTTPException(status_code=400, detail="Product ID already exists")
    products_db.append(product)
    return product

@router.get("/{product_id}", response_model=Product)
def get_product(product_id: int):
    for product in products_db:
        if product.id == product_id:
            return product
    raise HTTPException(status_code=404, detail="Product not found")