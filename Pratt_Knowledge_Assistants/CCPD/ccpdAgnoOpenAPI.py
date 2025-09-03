from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator, Union, Iterator
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager
import warnings

from pathlib import Path
from textwrap import dedent
from urllib.parse import quote_plus

from agno.agent import Agent
from agno.team.team import Team
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.markdown import MarkdownKnowledgeBase
from agno.tools.exa import ExaTools
from agno.models.openai import OpenAIChat, OpenAILike
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
career_advisor_team = None

# --- Lifespan Management for Agent Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global career_advisor_team
    logger.info("Initializing RAG agent on startup...")
    
    try:
        
        search_agent = Agent(
            tools=[ExaTools(
                num_results=2,
                show_results=True,
                text=False,
                highlights=False)],
            
            # model=OpenAILike(
            #     id="GLM-4.5-AIR", 
            #     base_url='http://172.19.50.61:8005/v1/', 
            #     api_key="empty",
            #     temperature=0.5
            # ),
            tool_call_limit=1,
            add_datetime_to_instructions=True,
            model=OpenAIChat(id="gpt-4.1"),
            show_tool_calls=False,
            markdown=True,
            description=(
                "You are ResearchBot-X, an expert at finding and extracting high-quality, "
                "up-to-date information from the web. Your job is to gather comprehensive, "
                "reliable, and diverse sources on the given topic."
            ),
            instructions=(
                "1. Search for the most recent and authoritative and up-to-date sources (news, blogs, official docs, research papers, forums, etc.) on the topic.\n"
                "2. Extract key facts, statistics, and expert opinions.\n"
                "3. Cover multiple perspectives and highlight any disagreements or controversies.\n"
                "4. Include relevant statistics, data, and expert opinions where possible.\n"
                "5. Organize your findings in a clear, structured format (e.g., markdown table or sections by source type).\n"
                "6. If the topic is ambiguous, clarify with the user before proceeding.\n"
                "7. Be as comprehensive and verbose as possible—err on the side of including more detail.\n"
                "8. Mention the References & Sources of the Content. (It's Must)"
            ),
        )
        
        pwd_escaped = quote_plus(db_password)
        db_url = f"postgresql+psycopg2://chatbot:{pwd_escaped}@127.0.0.1:5432/aichat"

        vector_db = PgVector(
            db_url=db_url,
            table_name="ccpd_docs",
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
            path="/home/nci/data/KnowledgeDocumentation/knowledge_base/ccpd_resources/_CCPD_Resources",
            vector_db=vector_db
        )
        kb.load(recreate=False)

        ccpd_agent = Agent(
            name="CCPD Assistant",
            role="""
Uses Pratt CCPD knowledge base to answer questions about career development resources. 
Specializes in career counseling, job search strategies, and professional development opportunities. 
Uses the vector database to search for relevant documents.
            """,
            # model=OpenAIChat(id="gpt-4.1"),
            model=OpenAILike(
                id="GLM-4.5-AIR", 
                base_url='http://172.19.50.61:8005/v1/', 
                api_key="empty",
                temperature=0.1
            ),
            instructions=dedent("""
Your mission is to provide comprehensive, authoritative support for students and users of the Pratt Center for Career and Professional Development (CCPD) resources, assisting them in navigating and understanding the content within various guides and materials. These resources include topics such as interview preparation, building portfolios, writing resumes, and other career development subjects. Follow the workflow below to ensure the best possible answer.

Analyze the Request
• Determine whether the user's question can be answered directly from existing knowledge or whether you must consult the CCPD guides and materials.
• Identify 1-3 concise search phrases that target specific career development concepts (e.g., "interview tips", "resume formatting", "portfolio design").
• Typical actions:
– Explaining a concept or strategy (no technical skills needed).
– Providing a step-by-step guide or best practices (may combine information from several sections of the CCPD resources).
• All search terms and topics are relevant to career and professional development as covered by CCPD materials.
After analysis, immediately begin the iterative search. Do not wait for user approval.

Iterative Search Process
• Use the search_knowledge_base tool to fetch related sections, instructions, and details from the CCPD guides and documentation (PDFs, handouts, or other resources).
• Keep refining or expanding your search terms until you either:
– Gather everything required to answer the question, or
– Exhaust all plausible search combinations.
Key CCPD topics you should be comfortable discussing:
• Interview preparation techniques and common questions
• Resume writing, formatting, and tailoring for specific industries
• Portfolio creation, organization, and presentation for creative fields
• Networking strategies and professional communication skills
• Job search tips, career planning, and professional development resources

Always deliver clear, step-by-step explanations, reference the exact sections or pathways in the CCPD guides, and cite the specific resources or materials you consulted.
If you do not know the answer, kindly ask the user to reach out to the Pratt Center for Career and Professional Development for further assistance.
"""),
            knowledge=kb,
            add_datetime_to_instructions=True,
            markdown=True,
        )
        
        career_advisor_team = Team(
            name="Student Career Support Team",
            mode="coordinate",
            model=OpenAILike(
                id="GLM-4.5-AIR", 
                base_url='http://172.19.50.61:8005/v1/', 
                api_key="empty",
                temperature=0.1
            ),
            enable_team_history=True,
            members=[search_agent, ccpd_agent],
            show_tool_calls=False,
            markdown=True,
            debug_mode=False,
            show_members_responses=False,
            success_criteria="Agents must provide accurate and relevant information based on the queries.",
            instructions=[
                "You are the student career development support agent responsible for classifying and routing student inquiries.", 
                "If the student query is not related to career development, politely inform the user that you can only assist with career-related queries and suggest they contact the appropriate department.",
                "Carefully analyze each user message and determine if it is: a question that needs research, or advice on resume, interview, career planning, or other career-related topics that can be answered directly from the knowledge base, or a request for an appointment or other services that require.",
                "For general questions about the career development, route to the ccpd_agent who will search documentation for answers.",
                "If the ccpd_agent cannot find an answer to a question, escalate it to the search_agent.",
                "For question related to graduate school applications, professional practice, or other specialized topics that require in-depth research, immediately route to the search_agent.",
                "Always provide a clear explanation of why you're routing the inquiry to a specific agent.",
                "After receiving a response from the appropriate agent, relay that information back to the user in a professional and helpful manner.",
                "Ensure a seamless experience for the user by maintaining context throughout the conversation.",
            ],
        )
        
        logger.info("RAG agent initialized successfully.")
        
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize agent: {e}", exc_info=True)
    
    yield
    logger.info("Shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="CCPD AI Assistant",
    description="AI assistant for CCPD documentation using OpenAI API standards.",
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
        # Format the conversation history as a single message
        conversation_parts = []
        for msg in messages:
            role = msg['role'].title()  # User, Assistant, System
            content = msg['content']
            conversation_parts.append(f"{role}: {content}")

        # Join all messages into a single query
        full_conversation = "\n\n".join(conversation_parts)
        
        response = career_advisor_team.run(message=full_conversation, markdown=True)
        
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
    if career_advisor_team is None:
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
                "id": "CCPD-AI-assistant",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "user",
            }
        ]
    }

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting CCPD RAG Assistant API...")
    uvicorn.run(
        "ccpdAgnoOpenAPI:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )