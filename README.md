# Pratt AI Embeddings Utilities

A collection of LangChain-compatible embedding utilities for various embedding APIs, developed by Ted Ngai at Pratt Institute.

## Overview

This repository contains Python utilities for working with different embedding APIs in a standardized, LangChain-compatible way. The utilities support both document embedding and query embedding, making them suitable for retrieval-augmented generation (RAG) applications, semantic search, and other AI workflows.

## Features

- **LangChain Compatible**: All embedding classes implement the LangChain `Embeddings` interface
- **Multiple API Support**: Support for Qwen and Infinity embedding servers
- **Matryoshka Embeddings**: Variable dimension embeddings for memory efficiency
- **Batch Processing**: Efficient batch processing with progress tracking
- **Dual-Mode Reranking**: Both standalone reranking and document compression (Infinity only)
- **DataFrame Integration**: Direct integration with pandas DataFrames
- **Health Monitoring**: Comprehensive API health checks and monitoring utilities
- **Session Management**: Optimized connection handling for Infinity server

## Supported APIs

### Qwen Embeddings (`QwenEmbeddings`)
- Compatible with Qwen embedding models (e.g., qwen3-embedding-8b)
- Support for both retrieval and clustering tasks
- **Matryoshka Support**: Variable dimension embeddings (512, 1024, 2048, etc.)
- **Batch Endpoint**: Efficient processing of queries and documents separately with `embed_batch()`
- **Advanced Search**: Built-in similarity search with `search_similar()` method
- **Dynamic Configuration**: Runtime dimension adjustment with `set_dimensions()` and `with_dimensions()`
- **API Discovery**: `get_available_models()`, `get_available_tasks()`, and `get_dimension_info()` methods

### Infinity Embeddings (`InfinityEmbeddingsReranker`)
- **Dual Interface**: Implements both `Embeddings` and `BaseDocumentCompressor` interfaces
- Compatible with Infinity server deployments
- Support for various open-source embedding models (BAAI/bge models, etc.)
- **Session-Based**: Persistent connection management for improved performance
- **Document Compression**: LangChain-compatible document reranking with `compress_documents()`
- **Flexible Reranking**: Direct reranking with `rank()` method, separate models for embeddings and reranking
- **Matryoshka Support**: Variable dimension embeddings with `dimensions` parameter
- **Advanced Features**: `embed_batch()`, `search_similar()`, and DataFrame integration
- **Optimized for Production**: Connection pooling and error recovery

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/pratt-ai-embeddings.git
cd pratt-ai-embeddings
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage with Qwen

```python
from util.qwen_embeddings import QwenEmbeddings

# Initialize embeddings client
embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    model_name="qwen3-embedding-8b",
    task="retrieval",
    dimensions=1024,  # Optional: Matryoshka dimension support
    show_progress=True
)

# Embed documents
documents = [
    "The capital of China is Beijing.",
    "Python is a programming language.",
    "Machine learning is a subset of AI."
]
doc_embeddings = embeddings.embed_documents(documents)

# Embed query
query = "What is the capital of China?"
query_embedding = embeddings.embed_query(query)

# Compute similarities
similarities = embeddings.compute_similarity([query_embedding], doc_embeddings)
print("Query-Document similarities:", similarities[0])
```

### Using with LangChain

```python
from langchain.vectorstores import FAISS
from util.qwen_embeddings import QwenEmbeddings

# Initialize embeddings
embeddings = QwenEmbeddings(api_url="http://localhost:8000")

# Create vector store
texts = ["Document 1", "Document 2", "Document 3"]
vectorstore = FAISS.from_texts(texts, embeddings)

# Search
results = vectorstore.similarity_search("your query", k=2)
```

### Reranking Documents

```python
# For reranking functionality, use InfinityEmbeddingsReranker
from util.infinityEmbedding import InfinityEmbeddingsReranker

# Initialize reranker
reranker = InfinityEmbeddingsReranker(
    api_url="http://localhost:8005",
    model_name="BAAI/bge-reranker-v2-m3",
    top_n=2
)

# Rerank documents for better relevance
query = "What is machine learning?"
documents = [
    "Machine learning is a method of data analysis.",
    "The weather is nice today.",
    "AI and ML are transforming industries."
]

reranked_results = reranker.rank(query=query, documents=documents)

for result in reranked_results:
    print(f"Score: {result['relevance_score']:.4f} - {result['document']}")
```

## Configuration

### Environment Variables

You can configure the embedding services using environment variables:

```bash
export QWEN_API_URL="http://localhost:8000"
export QWEN_MODEL_NAME="qwen3-embedding-4b"
export INFINITY_API_URL="http://localhost:8005"
```

### API Health Checks

```python
# Check if your embedding service is running
health_status = embeddings.health_check()
print("API Status:", health_status)

# Get available models
models = embeddings.get_available_models()
print("Available models:", models)
```

## Advanced Features

### DataFrame Integration

```python
import pandas as pd

# Embed text column in DataFrame
df = pd.DataFrame({
    'text': ['Document 1', 'Document 2', 'Document 3'],
    'metadata': ['Meta1', 'Meta2', 'Meta3']
})

embeddings_list = embeddings.embed_dataframe_column(df, 'text')

# Rerank DataFrame entries using InfinityEmbeddingsReranker
reranker = InfinityEmbeddingsReranker(
    api_url="http://localhost:8005",
    model_name="BAAI/bge-reranker-v2-m3"
)

reranked_results = reranker.rank(
    query="search query",
    documents=df['text'].tolist(),
    top_n=2
)

# Create reranked DataFrame
reranked_indices = [result['index'] for result in reranked_results]
reranked_df = df.iloc[reranked_indices].copy()
reranked_df['relevance_score'] = [result['relevance_score'] for result in reranked_results]
```

### Batch Processing with Progress

```python
# Process large document collections
large_document_list = [f"Document {i}" for i in range(1000)]

embeddings_with_progress = QwenEmbeddings(
    api_url="http://localhost:8000",
    batch_size=8,  # Updated default batch size
    show_progress=True  # Shows progress bar
)

all_embeddings = embeddings_with_progress.embed_documents(large_document_list)

# Or use the efficient batch endpoint
batch_result = embeddings_with_progress.embed_batch(
    queries=["query1", "query2"],
    documents=large_document_list[:100],  # Process first 100 docs
    dimensions=512  # Optional: override dimensions
)
```

## API Reference

### QwenEmbeddings

#### Parameters
- `api_url`: Base URL of the Qwen API server
- `model_name`: Model name (default: "qwen3-embedding-4b")
- `task`: Task type - "retrieval" or "clustering" (default: "retrieval")
- `dimensions`: Number of dimensions for Matryoshka embeddings (default: None)
- `batch_size`: Batch size for processing (default: 8)
- `show_progress`: Show progress bar (default: False)
- `timeout`: Request timeout in seconds (default: 300)

#### Methods
- `embed_documents(texts)`: Embed a list of documents
- `embed_query(text)`: Embed a single query
- `embed_batch(queries, documents, dimensions)`: Efficient batch processing
- `search_similar(query, documents, top_k, return_scores)`: Find similar documents
- `compute_similarity(embeddings1, embeddings2)`: Compute cosine similarity
- `embed_dataframe_column(df, column_name)`: Embed DataFrame column
- `get_available_models()`: Get list of available models
- `get_available_tasks()`: Get available embedding tasks
- `get_dimension_info()`: Get dimension information (Matryoshka)
- `health_check()`: Check API health status
- `set_dimensions(dimensions)`: Set dimensions for Matryoshka embeddings
- `with_dimensions(dimensions)`: Create new instance with different dimensions

### InfinityEmbeddingsReranker

#### Parameters
- `api_url`: Base URL of the Infinity server
- `model_name`: Model name for embeddings (default: None)
- `rerank_model_name`: Optional separate model name for reranking (default: None)
- `task`: Task type - "retrieval" or "clustering" (default: "retrieval")
- `dimensions`: Number of dimensions for Matryoshka embeddings (default: None)
- `top_n`: Number of top-ranking documents to return (default: 3)
- `batch_size`: Batch size for processing (default: 32)
- `show_progress`: Show progress bar (default: False)
- `timeout`: Request timeout in seconds (default: 300)

#### Methods
- `embed_documents(texts)`: Embed a list of documents (LangChain interface)
- `embed_query(text)`: Embed a single query (LangChain interface)
- `embed_batch(queries, documents, dimensions)`: Efficient batch processing
- `rank(query, documents)`: Direct reranking of documents
- `compress_documents(documents, query)`: LangChain document compression
- `search_similar(query, documents, top_k, return_scores)`: Find similar documents
- `compute_similarity(embeddings1, embeddings2)`: Compute cosine similarity
- `embed_dataframe_column(df, column_name)`: Embed DataFrame column
- `get_available_models()`: Get list of available models
- `health_check()`: Check API health status
- `set_dimensions(dimensions)`: Set dimensions for Matryoshka embeddings
- `with_dimensions(dimensions)`: Create new instance with different dimensions

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: Check the `util/README_embeddings.md` for detailed examples
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Contact**: Pratt Institute AI Team

## Changelog

### v2.0.0 (2025-08-04)
- **Major Update**: Separated embedding and reranking functionality
- **QwenEmbeddings**: Removed reranking methods, focused on embeddings only
- **InfinityEmbeddingsReranker**: New dual-interface class for both embeddings and reranking
- **Matryoshka Support**: Added variable dimension embeddings for both classes
- **New Methods**: Added `embed_batch()`, `search_similar()`, `get_available_tasks()`, `get_dimension_info()`
- **Updated Defaults**: Changed default model to "qwen3-embedding-4b", batch size to 8, timeout to 300s
- **Session Management**: Added persistent connection handling for Infinity server
- **Enhanced API**: Improved error handling and response validation

### v1.0.0 (2025-06-09)
- Initial release
- Support for Qwen and Infinity embedding APIs
- LangChain compatibility
- Reranking capabilities
- DataFrame integration
- Health monitoring utilities

---

**Developed by the Pratt Institute AI Team**
