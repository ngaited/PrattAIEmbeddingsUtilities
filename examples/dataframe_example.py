#!/usr/bin/env python3
"""
Example demonstrating DataFrame integration for batch processing.
"""

import pandas as pd
from util.qwen_embeddings import QwenEmbeddings


def dataframe_example():
    """Demonstrate DataFrame integration capabilities."""
    
    # Initialize embeddings client
    embeddings = QwenEmbeddings(
        api_url="http://localhost:8000",
        model_name="qwen3-embedding-8b",
        show_progress=True  # Show progress for batch processing
    )
    
    # Create sample dataset
    data = {
        'id': range(1, 21),
        'title': [
            "Introduction to Machine Learning",
            "Advanced Python Programming",
            "Data Visualization Techniques",
            "Natural Language Processing",
            "Computer Vision Fundamentals",
            "Deep Learning with Neural Networks",
            "Statistical Analysis Methods",
            "Big Data Processing",
            "Web Development with Flask",
            "Database Design Principles",
            "Artificial Intelligence Ethics",
            "Cloud Computing Platforms",
            "Cybersecurity Best Practices",
            "Software Engineering Patterns",
            "Mobile App Development",
            "Game Development Basics",
            "Blockchain Technology",
            "IoT Device Programming",
            "Quantum Computing Concepts",
            "Robotics and Automation"
        ],
        'description': [
            "Learn the fundamentals of machine learning algorithms and applications.",
            "Master advanced Python concepts including decorators, generators, and metaclasses.",
            "Create compelling visualizations using matplotlib, seaborn, and plotly.",
            "Process and analyze text data using NLP techniques and libraries.",
            "Understand image processing and computer vision algorithms.",
            "Build and train deep neural networks for various applications.",
            "Apply statistical methods to analyze and interpret data.",
            "Handle large datasets using distributed computing frameworks.",
            "Develop web applications using Flask framework and best practices.",
            "Design efficient and scalable database systems.",
            "Explore ethical considerations in AI development and deployment.",
            "Deploy applications on AWS, Azure, and Google Cloud platforms.",
            "Implement security measures to protect systems and data.",
            "Apply design patterns and principles in software development.",
            "Create mobile applications for iOS and Android platforms.",
            "Develop games using popular game engines and frameworks.",
            "Understand blockchain technology and cryptocurrency systems.",
            "Program Internet of Things devices and embedded systems.",
            "Explore quantum computing principles and quantum algorithms.",
            "Build autonomous robots and automation systems."
        ],
        'category': [
            'AI/ML', 'Programming', 'Data Science', 'AI/ML', 'AI/ML',
            'AI/ML', 'Data Science', 'Data Science', 'Web Dev', 'Database',
            'AI/ML', 'Cloud', 'Security', 'Software', 'Mobile',
            'Gaming', 'Blockchain', 'IoT', 'Quantum', 'Robotics'
        ],
        'difficulty': [
            'Beginner', 'Advanced', 'Intermediate', 'Intermediate', 'Intermediate',
            'Advanced', 'Intermediate', 'Advanced', 'Intermediate', 'Intermediate',
            'Beginner', 'Intermediate', 'Advanced', 'Intermediate', 'Intermediate',
            'Beginner', 'Advanced', 'Intermediate', 'Advanced', 'Advanced'
        ]
    }
    
    df = pd.DataFrame(data)
    print("Sample DataFrame:")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    
    # Embed the description column
    print("\nEmbedding course descriptions...")
    description_embeddings = embeddings.embed_dataframe_column(df, 'description')
    print(f"Created embeddings for {len(description_embeddings)} descriptions")
    
    # Search for similar documents using the search_similar method
    query = "I want to learn about artificial intelligence and machine learning"
    print(f"\nQuery: {query}")
    
    # Use search_similar to find relevant documents
    search_results = embeddings.search_similar(
        query=query,
        documents=df['description'].tolist(),
        top_k=5,
        return_scores=True
    )
    
    print("\nTop 5 most relevant courses:")
    print("-" * 80)
    for i, result in enumerate(search_results):
        idx = result['index']
        score = result['score']
        row = df.iloc[idx]
        print(f"Course: {row['title']}")
        print(f"Category: {row['category']} | Difficulty: {row['difficulty']}")
        print(f"Relevance Score: {score:.4f}")
        print(f"Description: {row['description'][:100]}...")
        print()
    
    # Filter by category and search
    ai_ml_courses = df[df['category'] == 'AI/ML'].copy()
    print(f"\nFiltering to AI/ML courses only ({len(ai_ml_courses)} courses):")
    
    ai_search_results = embeddings.search_similar(
        query="deep learning and neural networks",
        documents=ai_ml_courses['description'].tolist(),
        top_k=5,
        return_scores=True
    )
    
    print("\nAI/ML courses ranked by relevance to 'deep learning and neural networks':")
    for i, result in enumerate(ai_search_results):
        idx = result['index']
        score = result['score']
        row = ai_ml_courses.iloc[idx]
        print(f"- {row['title']} (Score: {score:.4f})")
    
    # Compare different queries
    queries = [
        "web development and APIs",
        "data analysis and statistics",
        "cybersecurity and protection"
    ]
    
    print("\n" + "="*80)
    print("Comparing different queries:")
    print("="*80)
    
    for query in queries:
        search_result = embeddings.search_similar(
            query=query,
            documents=df['description'].tolist(),
            top_k=1,
            return_scores=True
        )
        
        if search_result:
            top_result = search_result[0]
            idx = top_result['index']
            score = top_result['score']
            top_course = df.iloc[idx]
            
            print(f"\nQuery: '{query}'")
            print(f"Top course: {top_course['title']} (Score: {score:.4f})")


if __name__ == "__main__":
    dataframe_example()