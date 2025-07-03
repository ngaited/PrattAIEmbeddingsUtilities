from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown (if needed)
    # You can add cleanup code here
    
app = FastAPI(
    title="Qwen Embedding & Reranking API",
    description="OpenAI-compatible API for Qwen embeddings and reranking using transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Global model instances
embedding_model = None
embedding_tokenizer = None
reranking_model = None
reranking_tokenizer = None
device = None

# Task definitions
EMBEDDING_TASKS = {
    "retrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "clustering": "Identify and group similar text passages or documents based on their semantic content"
}

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract the last token embeddings with proper handling of padding"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction"""
    return f'Instruct: {task_description}\nQuery: {query}'

def format_rerank_instruction(instruction: str, query: str, doc: str) -> str:
    """Format instruction for reranking"""
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )

def load_models():
    """Load models on startup"""
    global embedding_model, embedding_tokenizer, reranking_model, reranking_tokenizer, device
    
    try:
        # Determine device - specifically use cuda:1
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
        
        try:
            # Try with flash attention first - initialize on CPU first
            embedding_model = AutoModel.from_pretrained(
                'Qwen/Qwen3-Embedding-8B', 
                attn_implementation="flash_attention_2", 
                torch_dtype=torch.float16,
                device_map={"": device}  # This ensures model loads directly to cuda:1
            )
            logger.info("Embedding model loaded with flash attention")
        except Exception as e:
            logger.warning(f"Flash attention failed, loading without: {e}")
            embedding_model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')
            embedding_model = embedding_model.to(device)
            logger.info("Embedding model loaded without flash attention")
        
        # Load reranking model
        logger.info("Loading reranking model...")
        reranking_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-8B", padding_side='left')
        
        try:
            # Try with flash attention first - initialize on CPU first
            reranking_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-Reranker-8B", 
                torch_dtype=torch.float16, 
                attn_implementation="flash_attention_2",
                device_map={"": device}  # This ensures model loads directly to cuda:1
            )
            reranking_model.eval()
            logger.info("Reranking model loaded with flash attention")
        except Exception as e:
            logger.warning(f"Flash attention failed for reranking, loading without: {e}")
            reranking_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-8B")
            reranking_model = reranking_model.to(device).eval()
            logger.info("Reranking model loaded without flash attention")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# OpenAI-compatible request/response models for embeddings
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Input text(s) to embed")
    model: str = Field(default="qwen3-embedding-8b", description="Model to use")
    task: Optional[Literal["retrieval", "clustering"]] = Field(default="retrieval", description="Task type")
    encoding_format: Optional[Literal["float", "base64"]] = Field(default="float", description="Encoding format")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions (not supported)")
    user: Optional[str] = Field(default=None, description="User identifier")

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

# Reranking request/response models
class RerankDocument(BaseModel):
    text: str = Field(..., description="Document text to rerank")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")

class RerankRequest(BaseModel):
    model: str = Field(default="qwen3-reranker-8b", description="Model to use")
    query: str = Field(..., description="Query text")
    documents: List[Union[str, RerankDocument]] = Field(..., description="Documents to rerank")
    task: Optional[str] = Field(default="Given a web search query, retrieve relevant passages that answer the query", description="Task description")
    top_n: Optional[int] = Field(default=None, description="Number of top results to return")
    return_documents: Optional[bool] = Field(default=True, description="Whether to return document texts")
    batch_size: int = Field(default=32, description="Batch size for reranking to control memory usage")

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[Union[str, RerankDocument]] = None

class RerankUsage(BaseModel):
    total_tokens: int

class RerankResponse(BaseModel):
    object: str = "list"
    results: List[RerankResult]
    model: str
    usage: RerankUsage

def process_rerank_inputs(pairs: List[str], tokenizer, max_length: int, prefix_tokens: List[int], suffix_tokens: List[int], device: torch.device):
    """Process inputs for reranking"""
    inputs = tokenizer(
        pairs, 
        padding=False, 
        truncation='longest_first',
        return_attention_mask=False, 
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(device) # Use the passed device
    
    return inputs

@torch.no_grad()
def compute_rerank_logits(inputs, model, token_true_id: int, token_false_id: int):
    """Compute reranking scores"""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings following OpenAI API standard with task support"""
    if embedding_model is None or embedding_tokenizer is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        # Handle both string and list inputs
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Get task description
        task_description = EMBEDDING_TASKS.get(request.task, EMBEDDING_TASKS["retrieval"])
        
        # Process texts based on task
        processed_texts = []
        for text in texts:
            if request.task in ["retrieval"]:
                # For retrieval, we assume single queries should use instruction
                # Multiple texts are treated as documents (no instruction needed)
                if len(texts) == 1:
                    processed_texts.append(get_detailed_instruct(task_description, text))
                else:
                    # For multiple texts, treat first as query, rest as documents
                    if text == texts[0]:
                        processed_texts.append(get_detailed_instruct(task_description, text))
                    else:
                        processed_texts.append(text)
            elif request.task == "clustering":
                # For clustering, no special instruction needed
                processed_texts.append(text)
            else:
                processed_texts.append(text)
        
        max_length = 8192
        
        # Tokenize the input texts
        batch_dict = embedding_tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = embedding_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Create response data
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(EmbeddingData(
                embedding=embedding.cpu().tolist(),
                index=i
            ))
        
        # Estimate token usage (rough approximation)
        total_tokens = sum(len(text.split()) for text in processed_texts)
        
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on query relevance using transformers"""
    if reranking_model is None or reranking_tokenizer is None:
        raise HTTPException(status_code=503, detail="Reranking model not loaded")
    
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents cannot be empty")
        
        doc_texts = [doc if isinstance(doc, str) else doc.text for doc in request.documents]
        
        # Setup reranking parameters
        token_false_id = reranking_tokenizer.convert_tokens_to_ids("no")
        token_true_id = reranking_tokenizer.convert_tokens_to_ids("yes")
        max_length = 8192
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        prefix_tokens = reranking_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = reranking_tokenizer.encode(suffix, add_special_tokens=False)
        
        all_scores = []
        
        # === BATCHING LOGIC START ===
        for i in range(0, len(doc_texts), request.batch_size):
            batch_doc_texts = doc_texts[i:i + request.batch_size]
            
            # Create formatted pairs for the current batch
            batch_pairs = [
                format_rerank_instruction(request.task, request.query, doc_text) 
                for doc_text in batch_doc_texts
            ]
            
            # Process inputs for the batch
            # Note: We need to get the model's device, especially when using device_map="auto"
            model_device = reranking_model.device
            inputs = process_rerank_inputs(batch_pairs, reranking_tokenizer, max_length, prefix_tokens, suffix_tokens, model_device)
            
            # Get relevance scores for the batch
            scores = compute_rerank_logits(inputs, reranking_model, token_true_id, token_false_id)
            all_scores.extend(scores)
            
            # Optional: Clear cache to free up memory between batches
            torch.cuda.empty_cache()
        # === BATCHING LOGIC END ===

        # Create results with scores and indices
        results_with_scores = []
        for i, score in enumerate(all_scores):
            result = RerankResult(
                index=i,
                relevance_score=float(score)
            )
            if request.return_documents:
                result.document = request.documents[i] # Use original documents list
            results_with_scores.append(result)
        
        # Sort by relevance score (descending)
        results_with_scores.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply top_n limit if specified
        if request.top_n is not None:
            results_with_scores = results_with_scores[:request.top_n]
        
        # Estimate token usage
        total_tokens = len(request.query.split()) + sum(len(text.split()) for text in doc_texts)
        
        return RerankResponse(
            results=results_with_scores,
            model=request.model,
            usage=RerankUsage(total_tokens=total_tokens)
        )
        
    except Exception as e:
        logger.error(f"Error reranking documents: {e}")
        # Add traceback for better debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reranking documents: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-embedding-8b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen",
                "permission": [],
                "root": "qwen3-embedding-8b",
                "parent": None,
            },
            {
                "id": "qwen3-reranker-8b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen",
                "permission": [],
                "root": "qwen3-reranker-8b",
                "parent": None,
            }
        ]
    }

@app.get("/v1/tasks")
async def list_tasks():
    """List available embedding tasks"""
    return {
        "object": "list",
        "embedding_tasks": EMBEDDING_TASKS
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model is not None,
        "reranking_model_loaded": reranking_model is not None,
        "device": str(device) if device else "unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)