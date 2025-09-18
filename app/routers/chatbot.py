from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from dataclasses import dataclass
import sqlite3
import hashlib
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    context: Optional[str] = Field(None, description="Additional context about the user's farm/location")
    crop_type: Optional[str] = Field(None, description="Specific crop user is asking about")

class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    sources: List[str]
    related_topics: List[str]
    response_type: str  # "rag", "llm", "hybrid"

@dataclass
class KnowledgeDocument:
    id: str
    content: str
    title: str
    category: str
    metadata: Dict[str, Any]

class AgricultureRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.knowledge_base: List[KnowledgeDocument] = []
        self.db_path = "agriculture_kb.db"
        self.embedding_dim = 384
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system components"""
        try:
            # Load sentence transformer model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector store
            self.vector_store = faiss.IndexFlatIP(self.embedding_dim)
            
            # Initialize database
            self.init_database()
            
            # Load knowledge base
            self.load_knowledge_base()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def init_database(self):
        """Initialize SQLite database for storing documents and metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_knowledge_base(self):
        """Load comprehensive agriculture knowledge base"""
        agriculture_knowledge = [
            {
                "title": "Crop Water Management",
                "content": """Proper water management is crucial for crop health and yield. Most crops require 1-1.5 inches of water per week, including rainfall. 
                Deep, infrequent watering is better than frequent shallow watering as it encourages deep root growth. 
                Monitor soil moisture at 6-8 inch depth. Drip irrigation systems can reduce water usage by 30-50% compared to sprinkler systems.
                Critical watering periods include seed germination, flowering, and fruit development stages.""",
                "category": "irrigation"
            },
            {
                "title": "Integrated Pest Management (IPM)",
                "content": """IPM combines biological, cultural, physical and chemical tools to minimize pest damage while protecting beneficial insects.
                Monitor pest populations weekly using yellow sticky traps and visual inspections. 
                Beneficial insects like ladybugs, lacewings, and parasitic wasps can control 60-90% of pest populations naturally.
                Neem oil is effective against aphids, whiteflies, and soft-bodied insects. Apply during cooler hours to avoid plant burn.
                Crop rotation breaks pest life cycles and reduces pesticide dependence.""",
                "category": "pest_control"
            },
            {
                "title": "Soil Health and Fertility",
                "content": """Healthy soil contains 25% air, 25% water, 45% minerals, and 5% organic matter.
                Optimal pH range for most crops is 6.0-7.5. Acidic soils (pH < 6.0) may need lime application.
                Organic matter improves soil structure, water retention, and nutrient availability.
                Soil testing should be done every 2-3 years to monitor nutrient levels and pH.
                Cover crops like crimson clover and winter rye prevent erosion and add nitrogen to soil.""",
                "category": "soil_management"
            },
            {
                "title": "Nutrient Management",
                "content": """Primary macronutrients (N-P-K) are needed in largest quantities. Nitrogen promotes leafy growth,
                phosphorus supports root development and flowering, potassium improves disease resistance and fruit quality.
                Secondary nutrients include calcium, magnesium, and sulfur. Micronutrients like iron, zinc, and boron
                are needed in small amounts but are essential for plant health.
                Compost provides slow-release nutrients and improves soil biology. A good compost contains 25-30 C:N ratio.""",
                "category": "nutrition"
            },
            {
                "title": "Climate and Weather Management",
                "content": """Understanding local climate patterns helps in crop selection and timing. Growing degree days (GDD)
                help predict plant development stages. Most crops have specific temperature ranges for optimal growth.
                Frost protection methods include row covers, water barrels for thermal mass, and wind machines.
                Extreme weather events are increasing due to climate change. Drought-resistant varieties and water conservation
                are becoming more important. Weather monitoring helps time irrigation, fertilization, and pest control.""",
                "category": "climate"
            },
            {
                "title": "Organic Farming Practices",
                "content": """Organic farming prohibits synthetic pesticides, herbicides, and fertilizers. Focus on soil health,
                biodiversity, and natural pest control. Composting, green manures, and biological pest control are key practices.
                Certification requires 3-year transition period. Organic crops often have premium market prices.
                Challenges include higher labor costs and potentially lower yields initially.""",
                "category": "organic_farming"
            },
            {
                "title": "Precision Agriculture Technology",
                "content": """GPS-guided tractors can reduce overlap and optimize field operations. Soil sensors provide real-time
                moisture and nutrient data. Drones can monitor crop health and identify problem areas early.
                Variable rate technology applies inputs based on specific field conditions. 
                Data analytics help optimize planting dates, irrigation scheduling, and harvest timing.""",
                "category": "technology"
            }
        ]
        
        # Convert to KnowledgeDocument objects and store
        embeddings = []
        for idx, doc in enumerate(agriculture_knowledge):
            doc_id = hashlib.md5(doc["content"].encode()).hexdigest()
            knowledge_doc = KnowledgeDocument(
                id=doc_id,
                content=doc["content"],
                title=doc["title"],
                category=doc["category"],
                metadata={"source": "internal_kb", "index": idx}
            )
            self.knowledge_base.append(knowledge_doc)
            
            # Generate embeddings
            embedding = self.embedding_model.encode([doc["content"]])[0]
            embeddings.append(embedding)
        
        # Add embeddings to vector store
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
            self.vector_store.add(embeddings_array)
            
        logger.info(f"Loaded {len(self.knowledge_base)} documents into knowledge base")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[tuple]:
        """Retrieve most relevant documents for a given query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search vector store
            scores, indices = self.vector_store.search(query_embedding, top_k)
            
            # Return documents with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_base):
                    results.append((self.knowledge_base[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_llm_response(self, query: str, context_docs: List[KnowledgeDocument]) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([f"Document: {doc.title}\nContent: {doc.content}" for doc in context_docs])
            
            # Create prompt for LLM
            prompt = f"""
            You are an expert agricultural advisor. Based on the following knowledge and the user's question, 
            provide a comprehensive, accurate, and practical answer.
            
            Context Information:
            {context}
            
            User Question: {query}
            
            Instructions:
            - Provide specific, actionable advice
            - Include relevant technical details
            - Mention any important considerations or warnings
            - If the context doesn't fully answer the question, say so clearly
            - Keep the response focused and practical
            """
            
            # Simulate LLM response (replace with actual OpenAI API call)
            # For demo purposes, we'll use a rule-based response generator
            response = self.simulate_llm_response(query, context_docs)
            
            return {
                "answer": response,
                "confidence": 0.85,  # This would come from the LLM
                "sources": [doc.title for doc in context_docs]
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing your query.",
                "confidence": 0.0,
                "sources": []
            }
    
    def simulate_llm_response(self, query: str, context_docs: List[KnowledgeDocument]) -> str:
        """Simulate LLM response based on retrieved context"""
        query_lower = query.lower()
        
        # Extract relevant information from context documents
        relevant_info = []
        for doc in context_docs:
            relevant_info.append(f"From {doc.title}: {doc.content[:200]}...")
        
        # Generate response based on query type
        if any(keyword in query_lower for keyword in ["water", "irrigation", "watering"]):
            return f"""Based on current agricultural best practices, here's what you need to know about watering:

{relevant_info[0] if relevant_info else 'Most crops require 1-1.5 inches of water per week, including rainfall.'}

Key recommendations:
• Monitor soil moisture at 6-8 inch depth
• Water deeply but less frequently to encourage root growth  
• Consider drip irrigation for water efficiency
• Adjust watering based on crop growth stage and weather conditions

Would you like specific advice for a particular crop or growing condition?"""
            
        elif any(keyword in query_lower for keyword in ["pest", "insect", "bug"]):
            return f"""For effective pest management, I recommend an Integrated Pest Management (IPM) approach:

{relevant_info[0] if relevant_info else 'IPM combines multiple strategies to control pests while minimizing environmental impact.'}

Action steps:
• Monitor weekly using visual inspections and sticky traps
• Encourage beneficial insects like ladybugs and parasitic wasps
• Use targeted treatments like neem oil for soft-bodied insects
• Practice crop rotation to break pest life cycles

What specific pest issues are you experiencing?"""
            
        elif any(keyword in query_lower for keyword in ["soil", "fertilizer", "nutrients"]):
            return f"""Soil health is fundamental to successful farming. Here's what you should know:

{relevant_info[0] if relevant_info else 'Healthy soil requires proper pH (6.0-7.5), adequate organic matter, and balanced nutrients.'}

Essential practices:
• Test soil every 2-3 years for pH and nutrients
• Add organic matter through compost or cover crops
• Balance N-P-K based on crop needs and soil test results
• Monitor soil structure and drainage

Do you have recent soil test results, or would you like guidance on soil testing?"""
            
        else:
            # General agricultural advice
            if relevant_info:
                return f"""Based on the available information: {relevant_info[0]}

This relates to your question about {query}. For the most accurate advice, I'd recommend:
• Consulting with your local agricultural extension office
• Considering your specific growing conditions and climate
• Testing any new practices on a small scale first

Could you provide more specific details about your farming situation for more targeted advice?"""
            else:
                return f"""Thank you for your question about {query}. While I don't have specific information readily available, 
here are some general recommendations:

• Consult your local agricultural extension service
• Connect with experienced farmers in your area  
• Consider soil and tissue testing for data-driven decisions
• Start with proven practices for your region and crop type

If you can provide more details about your specific situation, I'd be happy to give more targeted advice."""

# Initialize RAG system
rag_system = AgricultureRAGSystem()

@router.post("/query", response_model=ChatResponse)
async def advanced_chatbot_query(query: Query):
    """
    Advanced agricultural chatbot with RAG capabilities
    """
    try:
        start_time = datetime.now()
        
        # Retrieve relevant documents
        retrieved_docs = rag_system.retrieve_relevant_documents(query.question, top_k=3)
        
        if not retrieved_docs:
            # Fallback response
            return ChatResponse(
                answer="I don't have specific information about your question in my knowledge base. Could you provide more details or rephrase your question?",
                confidence_score=0.3,
                sources=[],
                related_topics=["general_farming", "crop_management"],
                response_type="fallback"
            )
        
        # Extract documents and scores
        docs, scores = zip(*retrieved_docs)
        avg_confidence = float(np.mean(scores))
        
        # Generate response using LLM with retrieved context
        llm_response = rag_system.generate_llm_response(query.question, list(docs))
        
        # Determine response type based on confidence
        response_type = "rag" if avg_confidence > 0.7 else "hybrid"
        
        # Generate related topics
        related_topics = list(set([doc.category for doc in docs]))
        
        # Store conversation in database (for analytics)
        try:
            conn = sqlite3.connect(rag_system.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (question, answer, confidence) VALUES (?, ?, ?)",
                (query.question, llm_response["answer"], avg_confidence)
            )
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.warning(f"Failed to store conversation: {db_error}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query processed in {processing_time:.2f}s with confidence {avg_confidence:.2f}")
        
        return ChatResponse(
            answer=llm_response["answer"],
            confidence_score=avg_confidence,
            sources=llm_response["sources"],
            related_topics=related_topics,
            response_type=response_type
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing your question")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system": "initialized",
        "knowledge_base_size": len(rag_system.knowledge_base),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/topics")
async def get_available_topics():
    """Get available topics in the knowledge base"""
    topics = list(set([doc.category for doc in rag_system.knowledge_base]))
    return {
        "available_topics": topics,
        "total_documents": len(rag_system.knowledge_base)
    }