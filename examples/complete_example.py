#!/usr/bin/env python3
"""
Example usage of Pratt AI Embeddings utilities.

This script demonstrates various features of the QwenEmbeddings and InfinityEmbeddingsReranker classes
including basic embeddings, reranking, DataFrame operations, and LangChain integration.
"""

import pandas as pd
from util.qwen_embeddings import QwenEmbeddings
from util.infinityEmbedding import InfinityEmbeddingsReranker


def main():
    """Main example function demonstrating all features."""
    
    # Initialize the Qwen embeddings client
    qwen_embeddings = QwenEmbeddings(
        api_url="http://localhost:8000",
        model_name="qwen3-embedding-8b",
        task="retrieval",
        show_progress=True
    )
    
    # Initialize the Infinity embeddings/reranker client with actual models
    infinity_embeddings = InfinityEmbeddingsReranker(
        api_url="http://localhost:8006",
        model_name="BAAI/bge-base-en-v1.5",
        rerank_model_name="Alibaba-NLP/gte-multilingual-reranker-base",
        top_n=3
    )
    
    # Check API health
    print("Qwen API Health Check:")
    print(qwen_embeddings.health_check())
    print()
    
    print("Infinity API Health Check:")
    print(infinity_embeddings.health_check())
    print()
    
    # Get available models
    print("Qwen available models:", qwen_embeddings.get_available_models())
    print("Infinity available models:", infinity_embeddings.get_available_models())
    print()
    
    # Example 1: Basic embedding usage (LangChain compatible)
    print("=== Example 1: Basic Embeddings ===")
    
    # Embed documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts objects.",
        "Python is a programming language."
    ]
    doc_embeddings = qwen_embeddings.embed_documents(documents)
    print(f"Embedded {len(documents)} documents")
    print(f"Embedding dimension: {len(doc_embeddings[0])}")
    
    # Embed query
    query = "What is the capital of China?"
    query_embedding = qwen_embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Compute similarities
    similarities = qwen_embeddings.compute_similarity([query_embedding], doc_embeddings)
    print("\nQuery-Document similarities:")
    for i, (doc, sim) in enumerate(zip(documents, similarities[0])):
        print(f"  Doc {i}: {sim:.4f} - {doc[:50]}...")
    print()
    
    # Example 2: Reranking with Infinity
    print("=== Example 2: Reranking ===")
    rerank_results = infinity_embeddings.rank(
        query="What is the capital of China?",
        documents=[
            "The capital of China is Beijing.",
            "Gravity is a force that attracts objects.",
            "Beijing is the political center of China.",
            "Python is a programming language.",
            "China's capital has a rich history."
        ]
    )
    
    print("Top 3 reranked results:")
    for i, result in enumerate(rerank_results[:infinity_embeddings.top_n]):
        print(f"  {i+1}. Score: {result['relevance_score']:.4f}")
        print(f"     Document: {result['document'][:80]}...")
        print(f"     Original index: {result['index']}")
    print()
    
    # Example 3: DataFrame operations
    print("=== Example 3: DataFrame Operations ===")
    
    # Create sample dataframe
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            "Beijing is the capital of China.",
            "The Great Wall is in China.",
            "Python is great for data science.",
            "Machine learning is fascinating.",
            "China has a long history."
        ],
        'category': ['geography', 'tourism', 'tech', 'tech', 'history']
    })
    
    # Embed dataframe column
    df_embeddings = qwen_embeddings.embed_dataframe_column(df, 'text')
    print(f"Embedded {len(df_embeddings)} texts from DataFrame")
    
    # Search similar documents using DataFrame content
    search_results = qwen_embeddings.search_similar(
        query="Tell me about China",
        documents=df['text'].tolist(),
        top_k=3,
        return_scores=True
    )
    
    print("\nTop 3 search results:")
    for i, result in enumerate(search_results):
        idx = result['index']
        score = result['score']
        row = df.iloc[idx]
        print(f"  {i+1}. Score: {score:.4f}")
        print(f"     ID: {row['id']}")
        print(f"     Text: {row['text']}")
        print(f"     Category: {row['category']}")
    print()
    
    # Example 4: Clustering task
    print("=== Example 4: Clustering Task ===")
    
    # Create embeddings client for clustering
    clustering_embeddings = QwenEmbeddings(
        api_url="http://localhost:8000",
        model_name="qwen3-embedding-8b",
        task="clustering"
    )
    
    clustering_texts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "The weather is sunny today.",
        "It might rain tomorrow.",
        "Python is used for AI development."
    ]
    
    cluster_embeddings = clustering_embeddings.embed_documents(clustering_texts)
    
    # Compute pairwise similarities
    similarity_matrix = clustering_embeddings.compute_similarity(cluster_embeddings, cluster_embeddings)
    
    print("Clustering similarity matrix:")
    for i, text in enumerate(clustering_texts):
        print(f"{i}: {text[:30]}...")
    print("\nSimilarity matrix:")
    print(similarity_matrix)
    
    # Example 5: Using with LangChain
    print("\n=== Example 5: LangChain Integration ===")
    
    try:
        # This shows it's compatible with LangChain's vector stores
        from langchain.schema import Document
        from langchain.vectorstores import FAISS
        
        # Create documents
        langchain_docs = [
            Document(page_content="Beijing is the capital of China.", metadata={"source": "geography"}),
            Document(page_content="Python is a programming language.", metadata={"source": "tech"}),
            Document(page_content="The Great Wall is in China.", metadata={"source": "tourism"})
        ]
        
        # Create vector store
        vector_store = FAISS.from_documents(langchain_docs, qwen_embeddings)
        
        # Search
        search_results = vector_store.similarity_search("What is in China?", k=2)
        print("LangChain vector store search results:")
        for i, doc in enumerate(search_results):
            print(f"  {i+1}. {doc.page_content} (source: {doc.metadata['source']})")
            
    except ImportError:
        print("LangChain and FAISS not installed. Skipping LangChain example.")
        print("To run this example, install: pip install langchain faiss-cpu")


if __name__ == "__main__":
    main()