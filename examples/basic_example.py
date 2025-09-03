#!/usr/bin/env python3
"""
Simple example demonstrating basic embedding functionality.
"""

from util.qwen_embeddings import QwenEmbeddings


def basic_embedding_example():
    """Simple example of document and query embedding."""
    
    # Initialize embeddings client
    embeddings = QwenEmbeddings(
        api_url="http://localhost:8000",
        model_name="qwen3-embedding-8b"
    )
    
    # Sample documents
    documents = [
        "Artificial intelligence is transforming industries.",
        "Machine learning helps computers learn from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information."
    ]
    
    # Embed documents
    print("Embedding documents...")
    doc_embeddings = embeddings.embed_documents(documents)
    print(f"Created embeddings for {len(documents)} documents")
    print(f"Embedding dimension: {len(doc_embeddings[0])}")
    
    # Query
    query = "What is machine learning?"
    query_embedding = embeddings.embed_query(query)
    
    # Compute similarities
    similarities = embeddings.compute_similarity([query_embedding], doc_embeddings)
    
    print(f"\nQuery: {query}")
    print("\nDocument similarities:")
    for i, (doc, sim) in enumerate(zip(documents, similarities[0])):
        print(f"{i+1}. Score: {sim:.4f} - {doc}")
    
    # Find most relevant document
    best_match_idx = similarities[0].argmax()
    print(f"\nMost relevant document: {documents[best_match_idx]}")


if __name__ == "__main__":
    basic_embedding_example()