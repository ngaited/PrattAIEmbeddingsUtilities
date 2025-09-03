# Examples

This directory contains example scripts demonstrating various features of the Pratt AI Embeddings utilities.

## Available Examples

### 1. `basic_example.py`
A simple introduction to document and query embedding functionality.

**Features demonstrated:**
- Basic document embedding
- Query embedding
- Similarity computation
- Finding the most relevant document

**Run with:**
```bash
python examples/basic_example.py
```

### 2. `reranking_example.py`
Demonstrates the reranking functionality for improved search results.

**Features demonstrated:**
- Document reranking based on query relevance
- Comparison between simple similarity and reranking
- Working with relevance scores
- Filtering top results

**Run with:**
```bash
python examples/reranking_example.py
```

### 3. `dataframe_example.py`
Shows how to integrate embeddings with pandas DataFrames for batch processing.

**Features demonstrated:**
- Embedding DataFrame columns
- Searching similar documents in DataFrames
- Batch processing with progress tracking
- Filtering and category-based analysis
- Multiple query comparisons

**Run with:**
```bash
python examples/dataframe_example.py
```

### 4. `complete_example.py`
Comprehensive example showing all available features.

**Features demonstrated:**
- All basic embedding operations
- Reranking with Infinity embeddings
- DataFrame integration
- Clustering task embeddings
- LangChain integration (if installed)
- API health checks and model listing

**Run with:**
```bash
python examples/complete_example.py
```

## Prerequisites

Before running the examples, make sure you have:

1. **API Servers Running**: 
   - Qwen API server at `http://localhost:8000`
   - Infinity API server at `http://localhost:8005`
   You can modify the `api_url` parameter in each script to point to your servers.

2. **Required Dependencies**: Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional Dependencies**: For the complete LangChain example:
   ```bash
   pip install langchain faiss-cpu
   ```

## Customizing Examples

### Changing API Configuration

To use a different API server or model, modify the initialization parameters:

```python
# For Qwen embeddings
embeddings = QwenEmbeddings(
    api_url="http://your-server:port",  # Change this
    model_name="your-model-name",       # Change this
    task="retrieval",                   # or "clustering"
    show_progress=True
)

# For Infinity embeddings/reranker
embeddings = InfinityEmbeddingsReranker(
    api_url="http://your-infinity-server:port",
    model_name="your-embedding-model",
    rerank_model_name="your-reranker-model"  # Optional
)
```

## Common Use Cases

### Semantic Search
Use the basic embedding example as a starting point for building semantic search functionality.

### Document Ranking
The reranking example shows how to improve search results by using specialized reranking models.

### Content Recommendation
The DataFrame example demonstrates how to build content recommendation systems using embeddings.

### RAG Applications
Combine these examples with vector databases for Retrieval-Augmented Generation applications.

## Troubleshooting

### Connection Issues
If you get connection errors, ensure your API servers are running and accessible:

```python
# Test API health
health = embeddings.health_check()
print(health)
```

### Memory Issues
For large datasets, use smaller batch sizes:

```python
embeddings = QwenEmbeddings(
    api_url="http://localhost:8000",
    batch_size=16,  # Reduce from default 8
    show_progress=True
)
```

### Performance Optimization
- Use batch processing for multiple documents
- Enable progress tracking for long operations
- Consider using clustering task for document grouping

## Next Steps

- Integrate with your own datasets
- Combine with vector databases like Pinecone, Weaviate, or Chroma
- Build full RAG applications using LangChain
- Experiment with different embedding models and tasks