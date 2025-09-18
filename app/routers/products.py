from fastapi import APIRouter
from pydantic import BaseModel
from typing import List


router = APIRouter()

# Product model
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: str = ""

# Fake DB
products_db = [
    Product(id=1, name="Wheat Seeds", price=100.0, description="High yield wheat"),
    Product(id=2, name="Rice Seeds", price=120.0, description="Premium quality rice"),
]

@router.get("/all", response_model=List[Product])
def get_all_products():
    return products_db