# Pratt AI Embeddings Utilities - Comprehensive Documentation

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Performance & Optimization](#performance--optimization)
- [Testing & Development](#testing--development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Pratt AI Embeddings Utilities is a comprehensive Python library developed by Ted Ngai at Pratt Institute that provides LangChain-compatible embedding utilities for various embedding APIs. This project serves as a unified interface for working with different embedding services, making it ideal for retrieval-augmented generation (RAG) applications, semantic search, and other AI workflows.

### Key Features

- **LangChain Compatibility**: All embedding classes implement the LangChain `Embeddings` interface
- **Multi-API Support**: Native support for Qwen and Infinity embedding servers
- **Batch Processing**: Efficient batch processing with progress tracking
- **Reranking Capabilities**: Built-in reranking for improved search relevance
- **DataFrame Integration**: Direct integration with pandas DataFrames
- **Health Monitoring**: Comprehensive API health checks and monitoring
- **Matryoshka Support**: Variable dimension embeddings for memory efficiency
- **Type Safety**: Full type hints and Pydantic validation

### Supported APIs

#### Qwen Embeddings (`QwenEmbeddings`)
- Compatible with Qwen embedding models (e.g., qwen3-embedding-8b, qwen3-embedding-4b)
- Support for both retrieval and clustering tasks
- Built-in reranking with qwen3-reranker-8b
- Matryoshka embedding support for variable dimensions
- Comprehensive similarity computation utilities

#### Infinity Embeddings (`InfinityEmbeddingsReranker`)
- Compatible with Infinity server deployments
- Support for various open-source embedding models
- Combined embedding and reranking functionality
- Optimized for local and self-hosted deployments
- Document compression capabilities

## Architecture

### Project Structure

```
Pratt AI Github/
├── util/                          # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── qwen_embeddings.py        # Qwen API implementation
│   └── infinityEmbedding.py      # Infinity API implementation
├── examples/                     # Usage examples
│   ├── basic_example.py          # Basic embedding usage
│   ├── complete_example.py       # Comprehensive feature demo
│   ├── dataframe_example.py      # DataFrame integration
│   └── reranking_example.py      # Reranking functionality
├── tests/                        # Test suite
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # Project overview
```

### Design Principles

1. **LangChain First**: All classes implement LangChain interfaces for seamless integration
2. **Consistent API**: Unified interface across different embedding providers
3. **Type Safety**: Comprehensive type hints and Pydantic models
4. **Error Handling**: Graceful error handling with informative messages
5. **Performance**: Optimized batch processing and connection management
6. **Extensibility**: Easy to add new embedding providers

### Class Hierarchy

```
Embeddings (LangChain)
├── QwenEmbeddings
│   ├── Basic embedding functionality
│   ├── Reranking capabilities
│   ├── DataFrame integration
│   └── Similarity computation
└── InfinityEmbeddingsReranker
    ├── Embeddings functionality
    ├── Document compression
    ├── Reranking as document compressor
    └── Session management
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Access to embedding API servers (Qwen or Infinity)
- Git (for development)

### Standard Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/pratt-ai-embeddings.git
cd pratt-ai-embeddings
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Development Installation

1. **Clone and setup**:
```bash
git clone https://github.com/your-username/pratt-ai-embeddings.git
cd pratt-ai-embeddings
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install development dependencies**:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### API Server Setup

#### Qwen API Server
```bash
# Example Qwen server setup
docker run -p 8000:8000 qwen-embedding-server:latest
```

#### Infinity Server
```bash
# Example Infinity server setup
docker run -p 8005:8000 infinity-emb-server:latest
```

## Core Components

### QwenEmbeddings Class

The primary class for interacting with Qwen embedding APIs.

#### Initialization
```python
from util.qwen_embeddings import QwenEmbeddings

embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    model_name="qwen3-embedding-8b",
    task="retrieval",  # or "clustering"
    dimensions=1024,   # Optional: Matryoshka dimensions
    batch_size=32,
    show_progress=True,
    timeout=300
)
```

#### Core Methods

##### embed_documents(texts: List[str]) -> List[List[float]]
Embed multiple documents for retrieval or clustering tasks.

```python
documents = [
    "The capital of China is Beijing.",
    "Python is a programming language.",
    "Machine learning is a subset of AI."
]
doc_embeddings = embeddings.embed_documents(documents)
```

##### embed_query(text: str) -> List[float]
Embed a single query for retrieval tasks.

```python
query = "What is the capital of China?"
query_embedding = embeddings.embed_query(query)
```

##### rerank(query, documents, top_n) -> List[Dict]
Rerank documents by relevance to a query.

```python
reranked_results = embeddings.rerank(
    query="What is machine learning?",
    documents=documents,
    top_n=3
)
```

### InfinityEmbeddingsReranker Class

A dual-purpose class providing both embedding and reranking functionality for Infinity servers.

#### Key Features
- Implements both `Embeddings` and `BaseDocumentCompressor` interfaces
- Session-based connection management
- Combined embedding and reranking workflows
- Document compression for LangChain pipelines

#### Initialization
```python
from util.infinityEmbedding import InfinityEmbeddingsReranker

reranker = InfinityEmbeddingsReranker(
    api_url="http://localhost:8005",
    model_name="BAAI/bge-large-en-v1.5",
    rerank_model_name="BAAI/bge-reranker-large",
    top_n=3,
    task="retrieval"
)
```

## API Reference

### QwenEmbeddings

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | str | Required | Base URL of the Qwen API server |
| `model_name` | str | "qwen3-embedding-4b" | Model name for embeddings |
| `task` | Literal["retrieval", "clustering"] | "retrieval" | Embedding task type |
| `dimensions` | Optional[int] | None | Matryoshka embedding dimensions |
| `batch_size` | int | 8 | Batch size for processing |
| `show_progress` | bool | False | Show progress bars |
| `timeout` | int | 300 | Request timeout in seconds |

#### Methods

##### Core Embedding Methods

**embed_documents(texts: List[str]) -> List[List[float]]**
- Embeds multiple documents
- Handles batch processing automatically
- Cleans text by replacing newlines with spaces
- Returns list of embedding vectors

**embed_query(text: str) -> List[float]**
- Embeds a single query
- Applies query-specific preprocessing for retrieval tasks
- Returns single embedding vector

**embed_batch(queries, documents, dimensions) -> Dict[str, Any]**
- Efficiently processes queries and documents separately
- Returns dictionary with embeddings and optional similarity scores
- Supports dimension override for specific calls

##### Reranking Methods

**rerank(query, documents, top_n, return_documents) -> List[Dict]**
- Reranks documents by relevance to query
- Returns list of results with relevance scores
- Supports custom top_n selection

**rerank_dataframe(query, df, text_column, top_n) -> pd.DataFrame**
- Reranks DataFrame entries based on text column
- Returns DataFrame with relevance scores added
- Preserves original DataFrame structure

##### Utility Methods

**compute_similarity(embeddings1, embeddings2) -> np.ndarray**
- Computes cosine similarity between embedding sets
- Returns similarity matrix
- Optimized for normalized embeddings

**embed_dataframe_column(df, column_name) -> List[List[float]]**
- Embeds text from DataFrame column
- Returns embeddings in DataFrame row order
- Handles missing columns gracefully

**search_similar(query, documents, top_k, return_scores) -> Union[List[str], List[Dict]]**
- Finds most similar documents to query
- Supports score return option
- Uses efficient batch processing

##### API Information Methods

**health_check() -> Dict[str, Any]**
- Checks API health status
- Returns health information or error details

**get_available_models() -> List[str]**
- Retrieves list of available models
- Returns model IDs or empty list on error

**get_available_tasks() -> Dict[str, str]**
- Gets supported embedding tasks
- Returns task descriptions

**get_dimension_info() -> Dict[str, Any]**
- Retrieves Matryoshka dimension information
- Returns dimension support details

### InfinityEmbeddingsReranker

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | str | Required | Base URL of the Infinity server |
| `model_name` | str | Required | Model name for embeddings |
| `rerank_model_name` | Optional[str] | None | Separate model for reranking |
| `top_n` | int | 3 | Number of top results for reranking |
| `task` | Literal["retrieval", "clustering"] | "retrieval" | Embedding task type |
| `dimensions` | Optional[int] | None | Matryoshka dimensions |
| `batch_size` | int | 32 | Batch size for processing |
| `show_progress` | bool | False | Show progress bars |
| `timeout` | int | 300 | Request timeout in seconds |

#### Methods

##### Document Compression (LangChain Integration)

**compress_documents(documents, query, callbacks) -> Sequence[Document]**
- Compresses document list by reranking
- Returns top_n most relevant documents
- Adds relevance scores to metadata
- Implements LangChain BaseDocumentCompressor interface

**rank(query, documents) -> List[Dict[str, Any]]**
- Direct reranking method
- Returns sorted results with scores
- Includes original document text

##### Session Management

The class maintains a requests Session for improved performance:
- Automatic connection pooling
- Persistent connections
- Configurable timeout handling

## Usage Examples

### Basic Usage

```python
from util.qwen_embeddings import QwenEmbeddings

# Initialize
embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    model_name="qwen3-embedding-8b"
)

# Embed documents
documents = [
    "Artificial intelligence is transforming industries.",
    "Machine learning helps computers learn from data.",
    "Deep learning uses neural networks."
]
doc_embeddings = embeddings.embed_documents(documents)

# Embed query
query = "What is machine learning?"
query_embedding = embeddings.embed_query(query)

# Compute similarities
similarities = embeddings.compute_similarity([query_embedding], doc_embeddings)
```

### LangChain Integration

```python
from langchain.vectorstores import FAISS
from langchain.schema import Document
from util.qwen_embeddings import QwenEmbeddings

# Initialize embeddings
embeddings = QwenEmbeddings(api_url="http://localhost:8000")

# Create documents
docs = [
    Document(page_content="Beijing is the capital of China."),
    Document(page_content="Python is a programming language."),
    Document(page_content="The Great Wall is in China.")
]

# Create vector store
vector_store = FAISS.from_documents(docs, embeddings)

# Search
results = vector_store.similarity_search("What is in China?", k=2)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### DataFrame Operations

```python
import pandas as pd
from util.qwen_embeddings import QwenEmbeddings

# Initialize
embeddings = QwenEmbeddings(show_progress=True)

# Create DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3],
    'text': [
        "Python is great for data science.",
        "Machine learning algorithms are powerful.",
        "Data visualization helps understand patterns."
    ],
    'category': ['tech', 'ai', 'data']
})

# Embed column
embeddings_list = embeddings.embed_dataframe_column(df, 'text')

# Rerank DataFrame
query = "Tell me about machine learning"
reranked_df = embeddings.rerank_dataframe(
    query=query,
    df=df,
    text_column='text',
    top_n=2
)

print("Top results:")
print(reranked_df[['id', 'text', 'rerank_score']])
```

### Reranking with Infinity

```python
from util.infinityEmbedding import InfinityEmbeddingsReranker
from langchain.schema import Document

# Initialize reranker
reranker = InfinityEmbeddingsReranker(
    api_url="http://localhost:8005",
    model_name="BAAI/bge-large-en-v1.5",
    rerank_model_name="BAAI/bge-reranker-large"
)

# Create documents
documents = [
    Document(page_content="Python is a programming language."),
    Document(page_content="The weather is nice today."),
    Document(page_content="Machine learning is a subset of AI.")
]

# Compress documents (rerank)
query = "What is artificial intelligence?"
compressed_docs = reranker.compress_documents(documents, query)

for doc in compressed_docs:
    print(f"Score: {doc.metadata['relevance_score']:.4f}")
    print(f"Content: {doc.page_content}")
```

### Batch Processing

```python
from util.qwen_embeddings import QwenEmbeddings

# Initialize with batch settings
embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    batch_size=64,
    show_progress=True
)

# Large document collection
large_documents = [f"Document {i}" for i in range(1000)]

# Process with progress bar
all_embeddings = embeddings.embed_documents(large_documents)

# Use batch endpoint for efficiency
batch_result = embeddings.embed_batch(
    queries=["What is AI?", "Tell me about Python"],
    documents=large_documents[:100]
)

print(f"Query embeddings: {len(batch_result['query_embeddings'])}")
print(f"Document embeddings: {len(batch_result['document_embeddings'])}")
```

## Advanced Features

### Matryoshka Embeddings

Support for variable-dimension embeddings for memory efficiency:

```python
# Initialize with specific dimensions
embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    dimensions=512  # Use only 512 dimensions instead of full 4096
)

# Or change dimensions dynamically
embeddings.set_dimensions(256)

# Create instance with different dimensions
compact_embeddings = embeddings.with_dimensions(128)
```

### Custom Tasks and Instructions

```python
# Custom reranking task
results = embeddings.rerank(
    query="artificial intelligence",
    documents=docs,
    task="Find documents specifically about AI applications in healthcare",
    top_n=5
)
```

### Health Monitoring

```python
# Comprehensive health check
health = embeddings.health_check()
print(f"API Status: {health.get('status', 'unknown')}")

# Get available models
models = embeddings.get_available_models()
print(f"Available models: {models}")

# Check dimension support
dim_info = embeddings.get_dimension_info()
print(f"Matryoshka support: {dim_info.get('supports_matryoshka', False)}")
```

### Error Handling and Retry Logic

```python
from util.qwen_embeddings import QwenEmbeddings
import time

def robust_embedding(embeddings, texts, max_retries=3):
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return []

# Usage
embeddings = QwenEmbeddings(api_url="http://localhost:8000")
try:
    embeddings_list = robust_embedding(embeddings, large_document_list)
except Exception as e:
    print(f"Failed after retries: {e}")
```

## Configuration

### Environment Variables

Configure the embedding services using environment variables:

```bash
# Qwen API Configuration
export QWEN_API_URL="http://localhost:8000"
export QWEN_MODEL_NAME="qwen3-embedding-8b"
export QWEN_RERANK_MODEL="qwen3-reranker-8b"

# Infinity API Configuration
export INFINITY_API_URL="http://localhost:8005"
export INFINITY_MODEL="BAAI/bge-large-en-v1.5"
export INFINITY_RERANK_MODEL="BAAI/bge-reranker-large"

# Performance Settings
export EMBEDDING_BATCH_SIZE=32
export EMBEDDING_TIMEOUT=300
export SHOW_PROGRESS=true
```

### Configuration File

Create a configuration file for complex setups:

```python
# config.py
from typing import Dict, Any

QWEN_CONFIG: Dict[str, Any] = {
    "api_url": "http://localhost:8000",
    "model_name": "qwen3-embedding-8b",
    "task": "retrieval",
    "dimensions": 1024,
    "batch_size": 32,
    "show_progress": True,
    "timeout": 300
}

INFINITY_CONFIG: Dict[str, Any] = {
    "api_url": "http://localhost:8005",
    "model_name": "BAAI/bge-large-en-v1.5",
    "rerank_model_name": "BAAI/bge-reranker-large",
    "top_n": 5,
    "task": "retrieval",
    "batch_size": 64,
    "show_progress": True
}

# Usage
from util.qwen_embeddings import QwenEmbeddings
from config import QWEN_CONFIG

embeddings = QwenEmbeddings(**QWEN_CONFIG)
```

### Model Selection Guide

| Use Case | Recommended Model | Dimensions | Notes |
|----------|------------------|------------|-------|
| General Retrieval | qwen3-embedding-8b | 4096 | Good balance of performance and quality |
| Memory-Constrained | qwen3-embedding-4b | 512-2048 | Use Matryoshka for flexibility |
| Multilingual | qwen3-embedding-8b | 4096 | Best for cross-lingual tasks |
| Clustering | qwen3-embedding-8b | 4096 | Use task="clustering" |
| Reranking | qwen3-reranker-8b | N/A | Specialized for relevance scoring |

## Performance & Optimization

### Batch Processing Optimization

```python
# Optimal batch sizes for different scenarios
BATCH_SIZE_RECOMMENDATIONS = {
    "small_documents": 64,    # Short texts (< 100 tokens)
    "medium_documents": 32,   # Medium texts (100-500 tokens)
    "large_documents": 16,    # Long texts (> 500 tokens)
    "memory_constrained": 8,  # Low memory environments
}

# Dynamic batch sizing
def get_optimal_batch_size(documents, memory_limit_gb=4):
    avg_length = sum(len(doc.split()) for doc in documents) / len(documents)
    
    if avg_length < 100:
        return BATCH_SIZE_RECOMMENDATIONS["small_documents"]
    elif avg_length < 500:
        return BATCH_SIZE_RECOMMENDATIONS["medium_documents"]
    else:
        return BATCH_SIZE_RECOMMENDATIONS["large_documents"]
```

### Memory Management

```python
# Memory-efficient processing for large datasets
def process_large_dataset(embeddings, documents, chunk_size=1000):
    """Process large datasets in chunks to manage memory."""
    all_embeddings = []
    
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        chunk_embeddings = embeddings.embed_documents(chunk)
        all_embeddings.extend(chunk_embeddings)
        
        # Optional: Save progress
        if i % 5000 == 0:
            print(f"Processed {i + len(chunk)} documents")
    
    return all_embeddings
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib

class CachedEmbeddings(QwenEmbeddings):
    @lru_cache(maxsize=1000)
    def embed_query_cached(self, text: str) -> List[float]:
        """Cached version of embed_query."""
        return self.embed_query(text)
    
    def get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for document lists."""
        combined = "|||".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def embed_documents_with_cache(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching support."""
        cache_key = self.get_cache_key(texts)
        # Implement your caching logic here
        return self.embed_documents(texts)
```

### Performance Monitoring

```python
import time
import psutil

class MonitoredEmbeddings(QwenEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0,
            "errors": 0
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = super().embed_documents(texts)
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += sum(len(text.split()) for text in texts)
            self.stats["total_time"] += time.time() - start_time
            
            return result
        except Exception as e:
            self.stats["errors"] += 1
            raise
        finally:
            end_memory = psutil.Process().memory_info().rss
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            print(f"Memory used: {memory_used:.2f} MB")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
```

## Testing & Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=util --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_qwen_embeddings.py

# Run with verbose output
pytest -v
```

### Test Structure

```python
# tests/test_qwen_embeddings.py
import pytest
from unittest.mock import Mock, patch
from util.qwen_embeddings import QwenEmbeddings

class TestQwenEmbeddings:
    @pytest.fixture
    def embeddings(self):
        return QwenEmbeddings(
            api_url="http://localhost:8000",
            model_name="test-model"
        )
    
    def test_embed_documents(self, embeddings):
        """Test document embedding."""
        documents = ["Test document 1", "Test document 2"]
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = embeddings.embed_documents(documents)
            
            assert len(result) == 2
            assert len(result[0]) == 3
            mock_post.assert_called_once()
    
    def test_embed_query(self, embeddings):
        """Test query embedding."""
        query = "Test query"
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = embeddings.embed_query(query)
            
            assert len(result) == 3
            assert isinstance(result, list)
    
    def test_compute_similarity(self, embeddings):
        """Test similarity computation."""
        embeddings1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embeddings2 = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]
        
        result = embeddings.compute_similarity(embeddings1, embeddings2)
        
        assert result.shape == (2, 2)
        assert isinstance(result, np.ndarray)
```

### Development Workflow

1. **Setup development environment**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

2. **Make changes**:
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# Run tests
pytest

# Run linting
black .
flake8 util/
mypy util/
```

3. **Submit changes**:
```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

### Code Quality Tools

The project uses several code quality tools:

- **Black**: Code formatting
- **flake8**: Linting and style checking
- **mypy**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality assurance

## Troubleshooting

### Common Issues

#### API Connection Errors

**Problem**: Connection refused or timeout errors

```python
# Solution: Check API health and configuration
embeddings = QwenEmbeddings(api_url="http://localhost:8000")

# Check health
health = embeddings.health_check()
if health.get("status") != "healthy":
    print(f"API unhealthy: {health}")

# Verify URL and port
import requests
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"API reachable: {response.status_code}")
except Exception as e:
    print(f"Connection error: {e}")
```

#### Memory Issues

**Problem**: Out of memory errors with large datasets

```python
# Solution: Use smaller batches and chunked processing
embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    batch_size=8,  # Reduce batch size
    dimensions=512  # Use smaller embeddings
)

# Process in chunks
def process_in_chunks(documents, chunk_size=100):
    all_embeddings = []
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        chunk_embeddings = embeddings.embed_documents(chunk)
        all_embeddings.extend(chunk_embeddings)
        # Clear memory
        del chunk_embeddings
    return all_embeddings
```

#### Rate Limiting

**Problem**: API rate limiting errors

```python
# Solution: Implement rate limiting and retry logic
import time
import random

def rate_limited_embed(embeddings, texts, max_retries=3):
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.random()
                print(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            else:
                raise
```

#### Model Compatibility

**Problem**: Model not found or incompatible model

```python
# Solution: Check available models
embeddings = QwenEmbeddings(api_url="http://localhost:8000")

# Get available models
models = embeddings.get_available_models()
print(f"Available models: {models}")

# Check if model is supported
if "qwen3-embedding-8b" not in models:
    print("Model not available, using fallback")
    embeddings.model_name = models[0] if models else "default"
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugEmbeddings(QwenEmbeddings):
    def _get_embeddings_batch(self, texts, is_query=False):
        logger.debug(f"Processing {len(texts)} texts, is_query={is_query}")
        logger.debug(f"Batch size: {self.batch_size}")
        logger.debug(f"Model: {self.model_name}")
        
        try:
            result = super()._get_embeddings_batch(texts, is_query)
            logger.debug(f"Successfully processed {len(result)} embeddings")
            return result
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
```

### Performance Profiling

Profile embedding performance:

```python
import cProfile
import pstats
from io import StringIO

def profile_embeddings():
    embeddings = QwenEmbeddings(api_url="http://localhost:8000")
    documents = ["Test document"] * 100
    
    # Profile the embedding process
    pr = cProfile.Profile()
    pr.enable()
    
    result = embeddings.embed_documents(documents)
    
    pr.disable()
    
    # Print profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

profile_embeddings()
```

## Contributing

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
```bash
git clone https://github.com/your-username/pratt-ai-embeddings.git
cd pratt-ai-embeddings
```

3. **Setup development environment**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

### Code Style Guidelines

- **Formatting**: Use Black for code formatting
- **Linting**: Follow flake8 guidelines
- **Type Hints**: Use comprehensive type hints
- **Docstrings**: Follow Google-style format
- **Testing**: Maintain high test coverage

### Commit Message Format

Use conventional commit format:

```
feat: add support for new embedding model
fix: resolve timeout issues in batch processing
docs: update API reference documentation
test: add unit tests for similarity computation
refactor: improve batch processing efficiency
```

### Pull Request Process

1. **Create feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes and test**:
```bash
# Make your changes
pytest
black .
flake8 util/
mypy util/
```

3. **Commit changes**:
```bash
git add .
git commit -m "feat: add your feature description"
```

4. **Push and create PR**:
```bash
git push origin feature/your-feature-name
```

### Areas for Contribution

- **New Embedding Providers**: Add support for additional APIs
- **Performance Optimizations**: Improve batch processing efficiency
- **Error Handling**: Enhance error messages and recovery
- **Documentation**: Add examples and tutorials
- **Testing**: Expand test coverage and add integration tests
- **Features**: Add new functionality like cross-encoder support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary

- **Commercial Use**: Allowed
- **Modification**: Allowed
- **Distribution**: Allowed
- **Private Use**: Allowed
- **Liability**: Provided "as is" without warranty
- **Attribution**: Must include license and copyright notice

### Third-Party Licenses

This project depends on several third-party libraries with their own licenses:

- **requests**: Apache 2.0
- **pandas**: BSD 3-Clause
- **tqdm**: MPL 2.0
- **langchain-core**: MIT
- **pydantic**: MIT
- **numpy**: BSD 3-Clause

---

**Developed by the Pratt Institute AI Team**  
*Last Updated: August 4, 2025*