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
logger = logging.getLogger("financial_advisor_chatbot")

# Load environment variables
load_dotenv()

# Check for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")

# Initialize the FastAPI app
app = FastAPI(title="Financial Advisor Chatbot")

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
vector_store_savings = None
vector_store_investments = None


# Load vector stores on startup
@app.on_event("startup")
async def startup_event():
    global vector_store_savings, vector_store_investments

    # Load savings plans vector store if it exists
    if os.path.exists("emb_savings_plans"):
        try:
            vector_store_savings = Chroma(
                persist_directory="emb_savings_plans",
                embedding_function=embeddings
            )
            logger.info("Savings plans vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading savings plans vector store: {str(e)}")
    else:
        logger.warning("Savings plans vector store directory not found")

    # Load investment opportunities vector store if it exists
    if os.path.exists("emb_default_english"):  # Using the existing real estate database for investments
        try:
            vector_store_investments = Chroma(
                persist_directory="emb_default_english",
                embedding_function=embeddings
            )
            logger.info("Investment opportunities vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading investment opportunities vector store: {str(e)}")
    else:
        logger.warning("Investment opportunities vector store directory not found")


# Function to detect if text contains Arabic
def is_arabic(text: str) -> bool:
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return bool(arabic_pattern.search(text))


# Function to determine if the user is interested in savings or investments
def determine_financial_goal(message: str) -> str:
    """
    Determine if the user is asking about savings or investments.
    """
    message = message.lower()

    # Keywords related to savings
    savings_keywords = ["save", "saving", "savings", "deposit", "emergency fund", "توفير", "ادخار", "وديعة"]

    # Keywords related to investments
    investment_keywords = ["invest", "investment", "property", "real estate", "stocks", "returns",
                           "استثمار", "عقار", "عقارات", "أسهم", "عوائد"]

    # Check for savings keywords
    for keyword in savings_keywords:
        if keyword in message:
            return "savings"

    # Check for investment keywords
    for keyword in investment_keywords:
        if keyword in message:
            return "investments"

    # Default to a general response if goal is unclear
    return "unknown"


# Function to get relevant financial products based on user query and goal
async def get_relevant_information(query: str, financial_goal: str, top_k: int = 5):
    """
    Retrieve the most relevant financial products based on the user query and goal.
    """
    if financial_goal == "savings" and vector_store_savings:
        try:
            docs = vector_store_savings.similarity_search(query, k=top_k)
            return docs, "savings"
        except Exception as e:
            logger.error(f"Error retrieving savings plans: {str(e)}")

    elif financial_goal == "investments" and vector_store_investments:
        try:
            docs = vector_store_investments.similarity_search(query, k=top_k)
            return docs, "investments"
        except Exception as e:
            logger.error(f"Error retrieving investment opportunities: {str(e)}")

    # If we couldn't determine the goal or don't have the right vector store
    if vector_store_investments:
        try:
            docs = vector_store_investments.similarity_search(query, k=top_k)
            return docs, "investments"
        except Exception as e:
            logger.error(f"Error retrieving default documents: {str(e)}")

    return [], "unknown"


# System prompts for the chatbot
SYSTEM_PROMPT_ENGLISH = """
You are a sophisticated financial advisor chatbot. Your goal is to understand users' financial objectives and recommend appropriate financial products based on their needs and preferences.

Start by asking questions to determine:
1. Whether they're interested in saving money or investing
2. Their financial goals and timeline (short-term, medium-term, long-term)
3. Their risk tolerance (conservative, moderate, aggressive)
4. Their budget and available funds (how much they can commit monthly/quarterly)
5. Any specific preferences or constraints

For SAVINGS goals:
- Recommend appropriate savings plans or accounts
- Explain interest rates, terms, and conditions
- Discuss liquidity and access to funds
- Highlight any tax advantages or special features

For INVESTMENT goals, especially in real estate:
- Recommend suitable properties based on their budget and goals
- Explain payment structures (monthly, quarterly, lump sum)
- Discuss expected returns and timelines
- Highlight location benefits, property features, and market trends
- Provide information on financing options if applicable

Keep your tone professional but approachable. Explain financial concepts clearly without jargon. Always respond to queries in English.

For product recommendations, format them clearly with the most important details:
- Product/Property name
- Key features and benefits
- Financial terms (costs, returns, payment schedules)
- Suitability for user's needs and goals
- Any special considerations or limitations

Present URLs as proper clickable links using markdown format: [Click here](URL)

Important: Include appropriate disclaimers about investment risks and encourage users to do their own research or consult with licensed financial advisors before making investment decisions.
"""

SYSTEM_PROMPT_ARABIC = """
أنت مستشار مالي متطور على شكل روبوت محادثة. هدفك هو فهم الأهداف المالية للمستخدمين والتوصية بالمنتجات المالية المناسبة بناءً على احتياجاتهم وتفضيلاتهم.

ابدأ بطرح أسئلة لتحديد:
1. ما إذا كانوا مهتمين بتوفير المال أو الاستثمار
2. أهدافهم المالية والإطار الزمني (قصير المدى، متوسط المدى، طويل المدى)
3. مدى تقبلهم للمخاطر (متحفظ، معتدل، جريء)
4. ميزانيتهم والأموال المتاحة (كم يمكنهم الالتزام به شهريًا/ربع سنويًا)
5. أي تفضيلات أو قيود محددة

بالنسبة لأهداف التوفير:
- اقترح خطط أو حسابات توفير مناسبة
- اشرح أسعار الفائدة والشروط والأحكام
- ناقش السيولة وإمكانية الوصول إلى الأموال
- سلط الضوء على أي مزايا ضريبية أو ميزات خاصة

بالنسبة لأهداف الاستثمار، خاصة في العقارات:
- اقترح عقارات مناسبة بناءً على ميزانيتهم وأهدافهم
- اشرح هياكل الدفع (شهري، ربع سنوي، دفعة واحدة)
- ناقش العوائد المتوقعة والجداول الزمنية
- سلط الضوء على فوائد الموقع وميزات العقار واتجاهات السوق
- قدم معلومات حول خيارات التمويل إذا كان ذلك مناسبًا

حافظ على نبرة مهنية ولكن ودية. اشرح المفاهيم المالية بوضوح دون مصطلحات تقنية. الرجاء الرد دائمًا على الاستفسارات باللغة العربية.

بالنسبة لتوصيات المنتجات، قم بتنسيقها بوضوح مع أهم التفاصيل:
- اسم المنتج/العقار
- الميزات والفوائد الرئيسية
- الشروط المالية (التكاليف، العوائد، جداول الدفع)
- مدى ملاءمته لاحتياجات المستخدم وأهدافه
- أي اعتبارات أو قيود خاصة

قدم الروابط كروابط قابلة للنقر باستخدام تنسيق ماركداون: [اضغط هنا](الرابط)

مهم: قم بتضمين إخلاء مسؤولية مناسب حول مخاطر الاستثمار وشجع المستخدمين على إجراء أبحاثهم الخاصة أو استشارة مستشارين ماليين مرخصين قبل اتخاذ قرارات استثمارية.

ستجد معلومات العقارات والمنتجات المالية باللغة الإنجليزية، قم بترجمتها إلى العربية في ردك.
"""


# Function to generate chatbot response
async def generate_response(user_message: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Generate a response to the user's message, taking into account the chat history.
    """
    # Determine if the query is in Arabic
    is_arabic_query = is_arabic(user_message)
    system_prompt = SYSTEM_PROMPT_ARABIC if is_arabic_query else SYSTEM_PROMPT_ENGLISH

    # Try to determine financial goal from message and chat history
    financial_goal = determine_financial_goal(user_message)

    # If goal is still unknown, try to infer from chat history
    if financial_goal == "unknown" and chat_history:
        history_text = " ".join([msg["content"] for msg in chat_history])
        financial_goal = determine_financial_goal(history_text)

    # Get relevant financial information based on determined goal
    relevant_info, goal_type = await get_relevant_information(user_message, financial_goal)

    # Format information for the context
    info_context = ""
    if relevant_info:
        if goal_type == "savings":
            info_context = "Here are some savings plans that might match the user's needs:\n\n"
        elif goal_type == "investments":
            info_context = "Here are some investment opportunities that might match the user's needs:\n\n"
        else:
            info_context = "Here is some financial information that might help answer the user's query:\n\n"

        for i, doc in enumerate(relevant_info, 1):
            if goal_type == "investments":
                info_context += f"Property {i}:\n{doc.page_content}\n\n"
            else:
                info_context += f"Option {i}:\n{doc.page_content}\n\n"
    else:
        info_context = "I don't have specific financial products matching this query in my database yet."

    # Add info about detected financial goal to the system prompt
    if financial_goal != "unknown":
        goal_info = f"\nThe user appears to be interested in {financial_goal}. "
        goal_info += "Prioritize recommendations and questions related to this goal."
        system_prompt += goal_info

    # Build messages for the chat model
    messages = [
        {"role": "system",
         "content": system_prompt + "\n\nContext information about available financial products:\n" + info_context}
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