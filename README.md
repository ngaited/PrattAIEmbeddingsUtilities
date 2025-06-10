# Pratt AI Embeddings Utilities

A collection of LangChain-compatible embedding utilities for various embedding APIs, developed at Pratt Institute.

## Overview

This repository contains Python utilities for working with different embedding APIs in a standardized, LangChain-compatible way. The utilities support both document embedding and query embedding, making them suitable for retrieval-augmented generation (RAG) applications, semantic search, and other AI workflows.

## Features

- **LangChain Compatible**: All embedding classes implement the LangChain `Embeddings` interface
- **Multiple API Support**: Support for Qwen and Infinity embedding servers
- **Batch Processing**: Efficient batch processing with progress tracking
- **Reranking Support**: Built-in reranking capabilities for improved search results
- **DataFrame Integration**: Direct integration with pandas DataFrames
- **Health Monitoring**: API health checks and monitoring utilities

## Supported APIs

### Qwen Embeddings (`QwenEmbeddings`)
- Compatible with Qwen embedding models (e.g., qwen3-embedding-8b)
- Support for both retrieval and clustering tasks
- Built-in reranking with qwen3-reranker-8b
- Comprehensive similarity computation utilities

### Infinity Embeddings (`InfinityEmbeddings`)
- Compatible with Infinity server deployments
- Support for various open-source embedding models
- Optimized for local and self-hosted deployments

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
# Rerank documents for better relevance
query = "What is machine learning?"
documents = [
    "Machine learning is a method of data analysis.",
    "The weather is nice today.",
    "AI and ML are transforming industries."
]

reranked_results = embeddings.rerank(
    query=query,
    documents=documents,
    top_n=2
)

for result in reranked_results:
    print(f"Score: {result['relevance_score']:.4f} - {result['document']['text']}")
```

## Configuration

### Environment Variables

You can configure the embedding services using environment variables:

```bash
export QWEN_API_URL="http://localhost:8000"
export QWEN_MODEL_NAME="qwen3-embedding-8b"
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

# Rerank DataFrame entries
reranked_df = embeddings.rerank_dataframe(
    query="search query",
    df=df,
    text_column='text',
    top_n=2
)
```

### Batch Processing with Progress

```python
# Process large document collections
large_document_list = [f"Document {i}" for i in range(1000)]

embeddings_with_progress = QwenEmbeddings(
    api_url="http://localhost:8000",
    batch_size=32,
    show_progress=True  # Shows progress bar
)

all_embeddings = embeddings_with_progress.embed_documents(large_document_list)
```

## API Reference

### QwenEmbeddings

#### Parameters
- `api_url`: Base URL of the Qwen API server
- `model_name`: Model name (default: "qwen3-embedding-8b")
- `rerank_model_name`: Reranker model (default: "qwen3-reranker-8b")
- `task`: Task type - "retrieval" or "clustering" (default: "retrieval")
- `batch_size`: Batch size for processing (default: 32)
- `show_progress`: Show progress bar (default: False)
- `timeout`: Request timeout in seconds (default: 30)

#### Methods
- `embed_documents(texts)`: Embed a list of documents
- `embed_query(text)`: Embed a single query
- `rerank(query, documents, top_n)`: Rerank documents by relevance
- `compute_similarity(embeddings1, embeddings2)`: Compute cosine similarity
- `health_check()`: Check API health status

### InfinityEmbeddings

#### Parameters
- `api_url`: Base URL of the Infinity server
- `model_name`: Model name for embeddings
- `batch_size`: Batch size for processing (default: 32)
- `show_progress`: Show progress bar (default: False)

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

### v1.0.0 (2025-06-09)
- Initial release
- Support for Qwen and Infinity embedding APIs
- LangChain compatibility
- Reranking capabilities
- DataFrame integration
- Health monitoring utilities

---

**Developed by the Pratt Institute AI Team**
