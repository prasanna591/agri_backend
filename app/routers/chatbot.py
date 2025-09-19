# filename: chatbot.py

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse, Response
import google.generativeai as genai
import os
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Deque
from dotenv import load_dotenv
import threading
from collections import defaultdict, deque
from pydantic import BaseModel, Field
import re
import base64

# --- Pydantic Models for Request and Response validation ---
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "en-US"

class ClearConversationRequest(BaseModel):
    session_id: str = "default"

# --- Create a Router instead of a full FastAPI app ---
router = APIRouter(
    prefix="/chatbot",  # This adds /chatbot before all endpoints defined here
    tags=["Chatbot"],    # Groups these endpoints in the API docs
)

# --- Logging (can be configured in your main app) ---
logger = logging.getLogger(__name__)

# --- RAG: Knowledge Base Loading ---
knowledge_base: List[Dict] = []

def load_knowledge_base():
    """Loads the knowledge base from the JSON file."""
    global knowledge_base
    try:
        # Assuming the knowledge_base is in a folder relative to this file
        kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base', 'disease_info.json')
        with open(kb_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        logger.info(f"Successfully loaded knowledge base with {len(knowledge_base)} entries.")
    except FileNotFoundError:
        logger.error("knowledge_base/disease_info.json not found. RAG will not be effective.")
        knowledge_base = []
    except json.JSONDecodeError:
        logger.error("Failed to decode knowledge_base/disease_info.json. Check for syntax errors.")
        knowledge_base = []

def retrieve_context(query: str) -> str:
    """
    Retrieves relevant context from the knowledge base based on the user's query.
    """
    if not knowledge_base:
        return "Knowledge base is not available."

    query_words = set(re.split(r'\s+', query.lower()))
    
    relevant_entries = []
    for category in knowledge_base:
        for item in category.get('items', []):
            searchable_text = ""
            if isinstance(item.get('name'), dict):
                searchable_text += item['name'].get('en', '') + " " + item['name'].get('ta', '')
            if isinstance(item.get('symptoms'), dict):
                searchable_text += " ".join(item['symptoms'].get('en', [])) + " " + " ".join(item['symptoms'].get('ta', []))
            
            if any(word in searchable_text.lower() for word in query_words):
                relevant_entries.append(json.dumps(item, ensure_ascii=False, indent=2))

    if not relevant_entries:
        return "No specific information found in the knowledge base for this query."
    
    return "\n---\n".join(relevant_entries[:3])


# --- Configuration (loaded from .env in your main app) ---


# --- Core Logic Classes (ConversationManager, GeminiAIHandler) - No Changes ---
class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self.message_counts: Dict[str, Deque] = defaultdict(lambda: deque())
        self.lock = threading.RLock()
        
    def get_conversation(self, session_id: str) -> List[Dict]:
        with self.lock:
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    'messages': [],
                    'created_at': datetime.now(),
                    'last_activity': datetime.now()
                }
            return self.conversations[session_id]['messages']
    
    def add_message(self, session_id: str, role: str, content: str):
        with self.lock:
            conversation = self.get_conversation(session_id)
            conversation.append({
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            })
            self.conversations[session_id]['last_activity'] = datetime.now()
            
            if len(conversation) > 20:
                self.conversations[session_id]['messages'] = conversation[-20:]
    
    def is_rate_limited(self, session_id: str) -> bool:
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            while (self.message_counts[session_id] and 
                   self.message_counts[session_id][0] < minute_ago):
                self.message_counts[session_id].popleft()
            
            if len(self.message_counts[session_id]) >= RATE_LIMIT_PER_MINUTE:
                return True
            
            self.message_counts[session_id].append(now)
            return False
    
    def cleanup_old_conversations(self):
        with self.lock:
            cutoff_time = datetime.now() - timedelta(seconds=CONVERSATION_TIMEOUT)
            expired_sessions = [
                session_id for session_id, data in self.conversations.items()
                if data['last_activity'] < cutoff_time
            ]
            
            for session_id in expired_sessions:
                del self.conversations[session_id]
                if session_id in self.message_counts:
                    del self.message_counts[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")

conversation_manager = ConversationManager()

class GeminiAIHandler:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.tts_model = genai.GenerativeModel("gemini-1.5-flash-preview-tts")
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    
    def generate_response(self, message: str, conversation_history: List[Dict] = None) -> Tuple[str, bool]:
        try:
            context = retrieve_context(message)
            
            history_str = "\n".join(
                [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in conversation_history[-6:]]
            )

            prompt = f"""
            You are an expert agricultural assistant for an e-commerce app. Your user might ask in English or Tamil.
            **Instructions:**
            1.  **Prioritize the Knowledge Base:** First, carefully analyze the "Retrieved Context from Knowledge Base" to answer the user's question. Base your answer strictly on this information if it's relevant.
            2.  **Fallback to General Knowledge:** If the retrieved context is not sufficient or doesn't contain the answer, then use your general knowledge to provide a helpful response.
            3.  **Language:** Respond in the same language as the user's question (detect if it's English or Tamil).
            4.  **Be Concise:** Provide clear, direct, and helpful answers.
            ---
            **Retrieved Context from Knowledge Base:**
            {context}
            ---
            **Recent Conversation History:**
            {history_str}
            ---
            **User's Current Question:** "{message}"
            ---
            **Assistant's Answer:**
            """
            
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config={"temperature": 0.7, "top_p": 0.8, "top_k": 40, "max_output_tokens": 2048}
            )
            
            if response and response.candidates:
                reply = response.candidates[0].content.parts[0].text
                return reply.strip(), True
            else:
                return "I couldn't generate a response. Please try again.", False
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return f"I encountered an error: {str(e)}", False

    def text_to_speech(self, text: str, lang: str) -> Optional[bytes]:
        try:
            response = self.tts_model.generate_content(
                f"Say in a clear and friendly voice: {text}",
                generation_config={"response_modalities": ["AUDIO"]},
            )

            if response and response.candidates:
                audio_part = next((part for part in response.candidates[0].content.parts if part.mime_type == "audio/wav"), None)
                if audio_part:
                     return audio_part.data
            return None
        except Exception as e:
            logger.error(f"Error in text-to-speech generation: {e}")
            return None

ai_handler = GeminiAIHandler()

# --- Background Task ---
def cleanup_worker():
    while True:
        time.sleep(300)
        try:
            conversation_manager.cleanup_old_conversations()
        except Exception as e:
            logger.error(f"Error in cleanup worker: {e}")

# --- Startup Function for your main.py to call ---
def start_chatbot_services():
    """This function should be called from your main app's startup event."""
    # The API Key and other env variables are guaranteed to be loaded here
    # since the main app's startup event has already run load_dotenv().
    # No need for this function to call load_dotenv() itself.
    load_dotenv() # <--- Keep this line here as a fallback

    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is missing!")

    genai.configure(api_key=API_KEY)

    # Now load other settings and the knowledge base
    global MAX_MESSAGE_LENGTH, RATE_LIMIT_PER_MINUTE, CONVERSATION_TIMEOUT
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "5000"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    CONVERSATION_TIMEOUT = int(os.getenv("CONVERSATION_TIMEOUT", "1800"))

    load_knowledge_base()
    logger.info("Starting Chatbot Services...")
    logger.info(f"Max message length: {MAX_MESSAGE_LENGTH}")
    logger.info(f"Rate limit: {RATE_LIMIT_PER_MINUTE} messages/minute")
    logger.info(f"Conversation timeout: {CONVERSATION_TIMEOUT} seconds")
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Chatbot background cleanup thread started.")

# --- API Endpoints defined on the router ---

@router.get("/")
def home():
    return {"status": "Chatbot module is running!"}

@router.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "active_conversations": len(conversation_manager.conversations)}

@router.post("/chat")
async def chat(chat_request: ChatRequest):
    start_time = time.time()
    
    message = chat_request.message.strip()
    session_id = chat_request.session_id
    
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Message too long. Maximum length is {MAX_MESSAGE_LENGTH} characters")
    
    if conversation_manager.is_rate_limited(session_id):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} messages per minute.")
    
    conversation_history = conversation_manager.get_conversation(session_id)
    conversation_manager.add_message(session_id, "user", message)
    
    reply, success = ai_handler.generate_response(message, conversation_history)
    
    if success:
        conversation_manager.add_message(session_id, "assistant", reply)
        response_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Chat request processed - Session: {session_id}, Response time: {response_time}ms")
        return {"reply": reply, "session_id": session_id, "response_time_ms": response_time, "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=reply)

@router.post("/text-to-speech")
async def text_to_speech(tts_request: TTSRequest):
    audio_data = ai_handler.text_to_speech(tts_request.text, tts_request.language)
    if audio_data:
        return Response(content=audio_data, media_type="audio/wav")
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate audio")

@router.get("/conversation/history")
def get_conversation_history(session_id: str = "default"):
    history = conversation_manager.get_conversation(session_id)
    return {"session_id": session_id, "messages": history, "count": len(history)}

@router.post("/conversation/clear")
def clear_conversation(clear_request: ClearConversationRequest):
    session_id = clear_request.session_id
    
    with conversation_manager.lock:
        if session_id in conversation_manager.conversations:
            del conversation_manager.conversations[session_id]
        if session_id in conversation_manager.message_counts:
            del conversation_manager.message_counts[session_id]
    
    logger.info(f"Conversation cleared for session: {session_id}")
    return {"message": "Conversation history cleared", "session_id": session_id}

@router.get("/stats")
def get_stats():
    with conversation_manager.lock:
        total_conversations = len(conversation_manager.conversations)
        total_messages = sum(len(conv['messages']) for conv in conversation_manager.conversations.values())
        
        return {
            "total_conversations": total_conversations, "total_messages": total_messages,
            "average_messages_per_conversation": round(total_messages / max(total_conversations, 1), 2)
        }