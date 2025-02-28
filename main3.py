import os
import re
from typing import List, Dict, Any, Optional
import logging

# Dependencies for LangChain
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# For Google's Gemini AI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# For the FastAPI web server
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("real_estate_chatbot")

# Load environment variables
load_dotenv()

# Check for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")

# Initialize the FastAPI app
app = FastAPI(title="Real Estate Broker Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Initialize Gemini Chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True
)

# Initialize vector stores from existing directories
vector_store_english = None
vector_store_arabic = None


# Load vector stores on startup
@app.on_event("startup")
async def startup_event():
    global vector_store_english, vector_store_arabic

    # Load English vector store if it exists
    if os.path.exists("emb_default_english"):
        try:
            vector_store_english = Chroma(
                persist_directory="emb_default_english",
                embedding_function=embeddings
            )
            logger.info("English vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading English vector store: {str(e)}")
    else:
        logger.warning("English vector store directory not found")

    # Load Arabic vector store if it exists
    if os.path.exists("emb_default_english"):
        try:
            vector_store_arabic = Chroma(
                persist_directory="eemb_default_english",
                embedding_function=embeddings
            )
            logger.info("Arabic vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Arabic vector store: {str(e)}")
    else:
        logger.warning("Arabic vector store directory not found")


# Function to detect if text contains Arabic
def is_arabic(text: str) -> bool:
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return bool(arabic_pattern.search(text))


# Function to get relevant properties based on user query
async def get_relevant_properties(query: str, top_k: int = 5):
    """
    Retrieve the most relevant properties based on the user query.
    """
    # Determine which vector store to use based on query language
    vector_store = vector_store_arabic if is_arabic(query) else vector_store_english

    if not vector_store:
        logger.warning(f"No vector store available for query language: {'Arabic' if is_arabic(query) else 'English'}")
        return []

    # Get the most relevant documents
    try:
        docs = vector_store.similarity_search(query, k=top_k)
        return docs
    except Exception as e:
        logger.error(f"Error retrieving properties: {str(e)}")
        return []


# System prompts for the chatbot
SYSTEM_PROMPT_ENGLISH = """
You are a helpful real estate broker chatbot. Your goal is to help users find properties that match their needs.
You have access to property information and can recommend suitable options.

First, ask questions to understand the user's preferences:
1. What type of property they're looking for (apartment, villa, office, etc.)
2. Whether they want to rent or buy
3. Their budget range
4. Preferred location(s)
5. Number of bedrooms/bathrooms needed
6. Any specific amenities or features they require

Based on their answers, you'll recommend properties that best match their requirements.
Keep conversations friendly and professional. If you don't have enough information,
politely ask for clarification. Always respond to queries in English.

For property recommendations, format them clearly with the most important details:
- Property title
- Type and status (rent/sale)
- Location
- Price
- Size and number of rooms
- Key features
- Brief description

If the user asks about a specific aspect of a property (like payment terms or viewing times),
provide the relevant information if available.
"""

SYSTEM_PROMPT_ARABIC = """
أنت روبوت دردشة وسيط عقاري مفيد. هدفك هو مساعدة المستخدمين في العثور على العقارات التي تناسب احتياجاتهم.
لديك إمكانية الوصول إلى معلومات العقارات ويمكنك التوصية بالخيارات المناسبة.

أولاً، اطرح أسئلة لفهم تفضيلات المستخدم:
1. نوع العقار الذي يبحثون عنه (شقة، فيلا، مكتب، إلخ)
2. ما إذا كانوا يريدون الإيجار أو الشراء
3. نطاق ميزانيتهم
4. الموقع/المواقع المفضلة
5. عدد غرف النوم/الحمامات المطلوبة
6. أي وسائل راحة أو ميزات محددة يحتاجونها

بناءً على إجاباتهم، ستوصي بالعقارات التي تتناسب بشكل أفضل مع متطلباتهم.
حافظ على المحادثات ودية ومهنية. إذا لم يكن لديك معلومات كافية،
اطلب التوضيح بأدب. الرجاء الرد دائمًا على الاستفسارات باللغة العربية.

بالنسبة لتوصيات العقارات، قم بتنسيقها بوضوح مع أهم التفاصيل:
- عنوان العقار
- النوع والحالة (إيجار/بيع)
- الموقع
- السعر
- الحجم وعدد الغرف
- الميزات الرئيسية
- وصف موجز

إذا سأل المستخدم عن جانب معين من العقار (مثل شروط الدفع أو أوقات المشاهدة)،
قدم المعلومات ذات الصلة إذا كانت متاحة.
"""


# Function to generate chatbot response
async def generate_response(user_message: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Generate a response to the user's message, taking into account the chat history.
    """
    # Determine if the query is in Arabic
    is_arabic_query = is_arabic(user_message)
    system_prompt = SYSTEM_PROMPT_ARABIC if is_arabic_query else SYSTEM_PROMPT_ENGLISH

    # Get relevant properties
    relevant_properties = await get_relevant_properties(user_message)

    # Format property information for the context
    property_context = ""
    if relevant_properties:
        property_context = "Here are some properties that might match what the user is looking for:\n\n"
        for i, doc in enumerate(relevant_properties, 1):
            property_context += f"Property {i}:\n{doc.page_content}\n\n"
    else:
        property_context = "I don't have specific properties matching this query in my database yet."

    # Build messages for the chat model
    messages = [
        {"role": "system",
         "content": system_prompt + "\n\nContext information about available properties:\n" + property_context}
    ]

    # Add chat history
    for msg in chat_history[-5:]:  # Only include the last 5 messages for context
        role = "assistant" if msg["role"] == "ai" else "user"
        messages.append({"role": role, "content": msg["content"]})

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    # Generate response
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error processing your request. Please try again."


# Redirect root to the static HTML interface
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


# API endpoint for health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Define a class for chat requests
class ChatRequest:
    def __init__(self, message: str, conversation_id: Optional[str] = None,
                 chat_history: Optional[List[Dict[str, str]]] = None):
        self.message = message
        self.conversation_id = conversation_id or "default"
        self.chat_history = chat_history or []


# API endpoint for chat
@app.post("/chat")
async def chat(request: Request):
    try:
        # Parse the request body
        data = await request.json()

        # Validate required fields
        if "message" not in data:
            raise HTTPException(status_code=400, detail="Missing required field: message")

        # Create a chat request object
        chat_request = ChatRequest(
            message=data["message"],
            conversation_id=data.get("conversation_id"),
            chat_history=data.get("chat_history", [])
        )

        # Generate response
        response = await generate_response(chat_request.message, chat_request.chat_history)

        # Return response
        return {
            "conversation_id": chat_request.conversation_id,
            "response": response
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Main function to run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)