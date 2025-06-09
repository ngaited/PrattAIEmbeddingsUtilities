'''
# example_usage.py
import pandas as pd
from qwen_embeddings import QwenEmbeddings

def main():
    # Initialize the embeddings client
    embeddings = QwenEmbeddings(
        api_url="http://localhost:8000",
        model_name="qwen3-embedding-8b",
        task="retrieval",  # or "clustering"
        show_progress=True
    )
    
    # Check API health
    print("API Health Check:")
    print(embeddings.health_check())
    print()
    
    # Get available models and tasks
    print("Available models:", embeddings.get_available_models())
    print("Available tasks:", embeddings.get_available_tasks())
    print()
    
    # Example 1: Basic embedding usage (LangChain compatible)
    print("=== Example 1: Basic Embeddings ===")
    
    # Embed documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts objects.",
        "Python is a programming language."
    ]
    doc_embeddings = embeddings.embed_documents(documents)
    print(f"Embedded {len(documents)} documents")
    print(f"Embedding dimension: {len(doc_embeddings[0])}")
    
    # Embed query
    query = "What is the capital of China?"
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Compute similarities
    similarities = embeddings.compute_similarity([query_embedding], doc_embeddings)
    print("\nQuery-Document similarities:")
    for i, (doc, sim) in enumerate(zip(documents, similarities[0])):
        print(f"  Doc {i}: {sim:.4f} - {doc[:50]}...")
    print()
    
    # Example 2: Reranking
    print("=== Example 2: Reranking ===")
    rerank_results = embeddings.rerank(
        query="What is the capital of China?",
        documents=[
            "The capital of China is Beijing.",
            "Gravity is a force that attracts objects.",
            "Beijing is the political center of China.",
            "Python is a programming language.",
            "China's capital has a rich history."
        ],
        top_n=3
    )
    
    print("Top 3 reranked results:")
    for i, result in enumerate(rerank_results):
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
    df_embeddings = embeddings.embed_dataframe_column(df, 'text')
    print(f"Embedded {len(df_embeddings)} texts from DataFrame")
    
    # Rerank dataframe
    df_reranked = embeddings.rerank_dataframe(
        query="Tell me about China",
        df=df,
        text_column='text',
        top_n=3
    )
    
    print("\nTop 3 reranked DataFrame results:")
    print(df_reranked[['id', 'text', 'rerank_score']])
    print()
    
    # Example 4: Clustering task
    print("=== Example 4: Clustering Task ===")
    
    # Create embeddings client for clustering
    clustering_embeddings = QwenEmbeddings(
        api_url="http://localhost:8000",
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
    
    # Example 5: Custom reranking task
    print("\n=== Example 5: Custom Reranking Task ===")
    
    custom_results = embeddings.rerank(
        query="artificial intelligence",
        documents=[
            "Neural networks are used in AI.",
            "The sun is shining brightly.",
            "Machine learning algorithms are powerful.",
            "I like pizza for dinner.",
            "Deep learning revolutionized AI."
        ],
        task="Find documents specifically about AI and machine learning technologies",
        top_n=3
    )
    
    print("Custom task reranking results:")
    for i, result in enumerate(custom_results):
        print(f"  {i+1}. Score: {result['relevance_score']:.4f} - {result['document'][:60]}...")

    # Example 6: Using with LangChain
    print("\n=== Example 6: LangChain Integration ===")
    
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
    vector_store = FAISS.from_documents(langchain_docs, embeddings)
    
    # Search
    search_results = vector_store.similarity_search("What is in China?", k=2)
    print("LangChain vector store search results:")
    for i, doc in enumerate(search_results):
        print(f"  {i+1}. {doc.page_content} (source: {doc.metadata['source']})")

if __name__ == "__main__":
    main()
'''