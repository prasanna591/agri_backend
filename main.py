from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import disease, chatbot, marketplace, orders, products

# Initialize FastAPI app
app = FastAPI(
    title="Agri Project Backend",
    description="API backend for agriculture app with e-commerce, chatbot, and crop disease detection",
    version="1.0.0"
)

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins (you can restrict to specific domains later)
    allow_credentials=True,
    allow_methods=["*"],       # allow all HTTP methods
    allow_headers=["*"],       # allow all headers
)

# Register routers
app.include_router(disease.router, prefix="/disease", tags=["Disease Detection"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["AI Chatbot"])
app.include_router(marketplace.router, prefix="/products", tags=["Marketplace"])
app.include_router(orders.router, prefix="/orders", tags=["Orders"])
app.include_router(products.router, prefix="/products", tags=["Products"])

# Root endpoint
@app.get("/")
def root():
    return {"message": "Agri Project Backend is running 🚀"}
