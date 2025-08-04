#!/usr/bin/env python3
"""
Example demonstrating reranking functionality for improved search results.
"""

from util.infinityEmbedding import InfinityEmbeddingsReranker


def reranking_example():
    """Demonstrate document reranking based on query relevance."""
    
    # Initialize embeddings client with actual models running on backend
    embeddings = InfinityEmbeddingsReranker(
        api_url="http://localhost:8006",
        model_name="BAAI/bge-base-en-v1.5",
        rerank_model_name="Alibaba-NLP/gte-multilingual-reranker-base",
        top_n=5
    )
    
    # Sample query
    query = "How to use Python for data science?"
    
    # Sample documents with varying relevance
    documents = [
        "Python is a versatile programming language used in many fields.",
        "Data science involves extracting insights from data using statistical methods.",
        "Python's pandas library is essential for data manipulation and analysis.",
        "Machine learning algorithms can be implemented using Python's scikit-learn.",
        "The weather forecast shows it will be sunny tomorrow.",
        "Jupyter notebooks provide an interactive environment for data science in Python.",
        "NumPy is the fundamental package for scientific computing in Python.",
        "Pizza is a popular Italian dish enjoyed worldwide.",
        "Matplotlib and seaborn are popular Python libraries for data visualization.",
        "Python's simplicity makes it ideal for rapid prototyping in data science projects."
    ]
    
    print(f"Query: {query}")
    print(f"\nReranking {len(documents)} documents...")
    
    # Rerank documents
    reranked_results = embeddings.rank(
        query=query,
        documents=documents
    )
    
    print(f"\nTop {embeddings.top_n} most relevant documents:")
    print("-" * 80)
    
    for i, result in enumerate(reranked_results[:embeddings.top_n]):
        score = result['relevance_score']
        document = result['document']
        original_index = result['index']
        
        print(f"{i+1}. Relevance Score: {score:.4f}")
        print(f"   Original Index: {original_index}")
        print(f"   Document: {document}")
        print()
    
    # Compare with simple similarity search
    print("=" * 80)
    print("Comparison with simple embedding similarity:")
    print("=" * 80)
    
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = embeddings.embed_documents(documents)
    similarities = embeddings.compute_similarity([query_embedding], doc_embeddings)
    
    # Get top 5 by similarity
    top_indices = similarities[0].argsort()[-5:][::-1]
    
    print("Top 5 by embedding similarity:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Similarity: {similarities[0][idx]:.4f}")
        print(f"   Document: {documents[idx]}")
        print()


if __name__ == "__main__":
    reranking_example()