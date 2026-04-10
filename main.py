import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv


load_dotenv()
# -----------------------------
# ENV VARIABLES (set in system)
# -----------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = "gpt-4o" # e.g. gpt-4o
API_VERSION = "2024-02-15-preview"

# -----------------------------
# FASTAPI INIT
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS (OPEN TO ALL)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ open to all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
# LLM SETUP (Azure OpenAI)
# -----------------------------
llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_version=API_VERSION,
    temperature=0.5,
)

# -----------------------------
# SYSTEM PROMPT (STRICT DOMAIN)
# -----------------------------
SYSTEM_PROMPT = """
You are an educational AI assistant built ONLY for a science exhibit.

Your topic is strictly:
"Hydroelectric power: from dam to electricity reaching homes"

You must ONLY answer questions related to:
- Hydroelectric dams
- Turbines, generators
- Transmission lines
- Step-up and step-down transformers
- Flow of electricity from dam to home

If the user asks anything outside this domain:
Respond strictly with:
"I am designed only for the hydroelectric dam exhibit. Please ask related questions."

Keep answers:
- Simple
- Educational
- Step-by-step when needed
- Easy for students to understand
"""

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=req.message)
        ]

        response = llm(messages)

        return {
            "response": response.content
        }

    except Exception as e:
        return {
            "error": str(e)
        }
    


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",       # file_name:app_instance
        host="0.0.0.0",
        port=8000,
        reload=True       # auto-restart on code changes
    )
