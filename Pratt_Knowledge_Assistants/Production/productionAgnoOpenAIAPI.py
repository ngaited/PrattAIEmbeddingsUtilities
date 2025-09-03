from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager
import warnings

from pathlib import Path
from textwrap import dedent
from urllib.parse import quote_plus

from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.markdown import MarkdownKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.models.litellm import LiteLLM
from agno.vectordb.pgvector import PgVector, SearchType
from agno.reranker.infinity import InfinityReranker

import os
import logging
import dotenv

dotenv.load_dotenv()
warnings.filterwarnings("ignore")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables ---
openai_key = os.getenv("OPENAI_API_KEY")
db_password = os.getenv("db_password")

# --- Global Agent Variable ---
agent = None

# --- Lifespan Management for Agent Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Initializing RAG agent on startup...")
    
    try:
        pwd_escaped = quote_plus(db_password)
        db_url = f"postgresql+psycopg2://chatbot:{pwd_escaped}@127.0.0.1:5432/aichat"

        vector_db = PgVector(
            db_url=db_url,
            table_name="production_docs",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
            reranker=InfinityReranker(
                model="Alibaba-NLP/gte-multilingual-reranker-base",  # You can change this to other models
                host="http://172.16.32.24",
                port=8006,
                top_n=5,  # Return top 5 reranked documents
            ),
        )

        kb = MarkdownKnowledgeBase(
            path="/home/nci/data/KnowledgeDocumentation/knowledge_base/Production_Knowledge_Base",
            vector_db=vector_db
        )
        kb.load(recreate=False)

        agent = Agent(
            name="Production Assistant",
            model=LiteLLM(id="gpt-4.1", temperature=0.0),
            instructions=dedent("""\
    You are a RAG agent with a mission to provide comprehensive, authoritative support for users of Pratt Institute's production facilities, equipment, and tools.
    Follow the workflow below to guarantee the best possible answer. Only use information from Pratt's production knowledge base, which includes detailed documentation on all production equipment and procedures.
    Do not use any external sources or general knowledge about production equipment.If you don't know the answer, say "I don't know" instead of making up information.

    1. **Analyse the request**
    • Decide whether the user's question can be answered directly from existing knowledge about Pratt's production facilities and equipment.
    • Identify the specific machines or tools the user is inquiring about (e.g., "digital cutter", "3D printer", "laser cutter").
    • Typical actions:
        – Explaining how a specific machine works.
        – Supplying step-by-step procedures for using equipment.
        – Troubleshooting common issues with production tools.
    • All terminology, procedures, and examples should be specific to Pratt Institute's production facilities.

    After analysis, immediately begin providing helpful information based on your knowledge of Pratt's production equipment.

    2. **Comprehensive Support Process**
    • Provide detailed information about machine operation, safety protocols, and best practices.
    • Include specific settings, materials compatibility, and technical specifications when relevant.
    • Offer alternative approaches or equipment recommendations when appropriate.

    Key production equipment and topics you should be comfortable discussing:
    • 3D printers (operation, filament types, slicing software, troubleshooting)
    • Laser cutters (power settings, compatible materials, vector vs raster)
    • Digital cutters (blade types, material preparation, software)
    • 2D printing equipment (paper types, color calibration, file preparation)
    • Water jet cutters (material thickness, pressure settings, safety)
    • CNC machines (toolpaths, material mounting, feed rates)
    • Workshop tools and equipment (proper usage and safety)
    • Material selection and preparation for various processes

    Always deliver clear, step-by-step explanations, reference specific machine models when possible, and prioritize safety instructions when applicable. If you don't know the answer, say "I don't know" instead of making up information.
    """),
            knowledge=kb,
            add_datetime_to_instructions=True,
            markdown=True,
        )
        logger.info("RAG agent initialized successfully.")
        
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize agent: {e}", exc_info=True)
    
    yield
    logger.info("Shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Production RAG Assistant API",
    description="A RAG-powered assistant for Production documentation using OpenAI API standards.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI API Compatible Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

class StreamingDelta(BaseModel):
    content: Optional[str] = None

class StreamingChoice(BaseModel):
    index: int
    delta: StreamingDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamingChoice]

# --- Core Agent Logic ---

def get_agent_response(messages: List[Dict[str, str]]) -> str:
    """
    Get a complete response from the Agno agent.
    Accepts a list of message dicts to maintain conversation history.
    """
    try:
        response = agent.run(messages=messages)
        
        if hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'messages') and response.messages:
            return response.messages[-1].content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    except Exception as e:
        logger.error(f"Error in get_agent_response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def generate_streaming_response(messages: List[Dict[str, str]], response_id: str, model: str) -> AsyncGenerator[str, None]:
    """
    Generate streaming response chunks.
    1. Immediately sends a "thinking" message.
    2. Runs the blocking RAG agent in a separate thread.
    3. Streams the final answer once it's ready.
    """
    try:
        # --- Step 1: Immediately send the "thinking" tag ---
        # This is sent to the front-end right away, providing instant feedback.
        # The brackets and italics are just for styling, you can customize this.
        thinking_content = "[*Searching knowledge base...*]\n\n"
        thinking_chunk = ChatCompletionChunk(
            id=response_id,
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[StreamingChoice(index=0, delta=StreamingDelta(content=thinking_content))]
        )
        yield f"data: {thinking_chunk.model_dump_json()}\n\n"
        logger.info("Sent 'thinking' tag to client.")

        # --- Step 2: Run the blocking agent call in a background thread ---
        # asyncio.to_thread runs the synchronous get_agent_response function
        # in a separate thread, preventing it from blocking the main server loop.
        # The 'await' here means we wait for the thread to finish before proceeding.
        full_response = await asyncio.to_thread(get_agent_response, messages)
        logger.info("Agent processing finished, received final response.")

        # --- Step 3: Stream the final response ---
        # Now that we have the full_response, we stream it to the client.
        # Open WebUI will append this to the "thinking" message it already displayed.
        chunk_size = 20
        for i in range(0, len(full_response), chunk_size):
            chunk_content = full_response[i:i+chunk_size]
            chunk = ChatCompletionChunk(
                id=response_id,
                created=int(datetime.now().timestamp()),
                model=model,
                choices=[StreamingChoice(index=0, delta=StreamingDelta(content=chunk_content))]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.01) # A small delay for a smoother typing effect

        # --- Step 4: Send the final chunk to signal completion ---
        final_chunk = ChatCompletionChunk(
            id=response_id,
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[StreamingChoice(index=0, delta=StreamingDelta(), finish_reason="stop")]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {e}", exc_info=True)
        error_content = f"\n\nAn error occurred while generating the response: {str(e)}"
        error_chunk = ChatCompletionChunk(
            id=response_id,
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[StreamingChoice(index=0, delta=StreamingDelta(content=error_content), finish_reason="stop")]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

# --- API Endpoints ---

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Please try again shortly.")

    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(datetime.now().timestamp())
    
    message_dicts = [msg.model_dump() for msg in request.messages]
    logger.info(f"Received {len(message_dicts)} messages. Last message: '{message_dicts[-1]['content'][:100]}...'")

    if request.stream:
        return StreamingResponse(
            generate_streaming_response(message_dicts, response_id, request.model),
            media_type="text/event-stream"
        )
    else:
        answer = get_agent_response(messages=message_dicts)
        
        response = ChatCompletionResponse(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=answer),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": sum(len(m['content']) for m in message_dicts),
                "completion_tokens": len(answer),
                "total_tokens": sum(len(m['content']) for m in message_dicts) + len(answer)
            }
        )
        return response

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Assembly-Loop-Lap-AI-assistant",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "user",
            }
        ]
    }

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Production Assistant API...")
    uvicorn.run(
        "productionAgnoOpenAIAPI:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )