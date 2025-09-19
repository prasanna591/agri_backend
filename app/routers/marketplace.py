from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(prefix="/products", tags=["products"])

# --- Models ---
class Product(BaseModel):
    """
    Represents a product in the inventory.
    """
    id: int = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Name of the product")
    price: float = Field(..., gt=0, description="Price of the product (must be greater than 0)")
    description: Optional[str] = Field(None, description="Detailed description of the product")

# --- In-memory product store ---
products_db: List[Product] = [
    Product(id=1, name="Organic Fertilizer", price=299.0, description="Rich in nitrogen and potassium."),
    Product(id=2, name="Hybrid Seeds Pack", price=149.0, description="High-yield, drought-resistant."),
    Product(id=3, name="Drip Irrigation Kit", price=899.0, description="Water-efficient irrigation system."),
]

# --- Helper function for finding a product ---
def find_product(product_id: int):
    """
    Finds a product by its ID in the in-memory database.
    Returns the product object if found, otherwise None.
    """
    return next((product for product in products_db if product.id == product_id), None)

# --- Endpoints ---
@router.get("/", response_model=List[Product], summary="Get all products")
def get_all_products():
    """
    Retrieves a list of all products currently in the inventory.
    """
    return products_db

@router.get("/{product_id}", response_model=Product, summary="Get a product by ID")
def get_product_by_id(product_id: int):
    """
    Retrieves a single product using its unique ID.
    Raises a 404 error if the product is not found.
    """
    product = find_product(product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found"
        )
    return product

@router.post("/", response_model=Product, status_code=status.HTTP_201_CREATED, summary="Add a new product")
def add_new_product(product: Product):
    """
    Adds a new product to the inventory.
    Prevents adding a product with an ID that already exists.
    """
    if find_product(product.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Product with ID {product.id} already exists"
        )
    products_db.append(product)
    return product

@router.put("/{product_id}", response_model=Product, summary="Update an existing product")
def update_existing_product(product_id: int, updated_product: Product):
    """
    Updates an existing product's details.
    Raises a 404 error if the product to be updated is not found.
    """
    product_index = -1
    for i, product in enumerate(products_db):
        if product.id == product_id:
            product_index = i
            break
            
    if product_index == -1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found"
        )
    
    # Ensure the ID in the request body matches the path parameter ID
    if updated_product.id != product_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID in path and request body must match"
        )

    products_db[product_index] = updated_product
    return updated_product

@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a product")
def delete_product(product_id: int):
    """
    Deletes a product from the inventory using its ID.
    Raises a 404 error if the product to be deleted is not found.
    """
    global products_db
    initial_length = len(products_db)
    products_db = [product for product in products_db if product.id != product_id]
    
    if len(products_db) == initial_length:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found"
        )
    # The 204 status code means no content should be returned
    return