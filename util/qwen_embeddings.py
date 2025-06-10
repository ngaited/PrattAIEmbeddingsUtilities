# qwen_embeddings.py
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Union, Literal
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, computed_field
import numpy as np


class QwenEmbeddings(BaseModel, Embeddings):
    """
    A LangChain-compatible class to get embeddings from the Qwen API server.
    
    Example:
        .. code-block:: python

            from qwen_embeddings import QwenEmbeddings

            embeddings = QwenEmbeddings(
                api_url="http://localhost:8000",
                model_name="qwen3-embedding-8b"
            )
            
            # Use with LangChain
            docs = ["Hello world", "Another document"]
            doc_embeddings = embeddings.embed_documents(docs)
            
            query = "What is this about?"
            query_embedding = embeddings.embed_query(query)
            
            # Use reranking
            reranked = embeddings.rerank(
                query="What is the capital?",
                documents=["Beijing is the capital.", "Python is a language."],
                top_n=1
            )
    """
    
    api_url: str = Field(..., description="The base URL of the Qwen API server")
    model_name: str = Field(default="qwen3-embedding-8b", description="The model name to use for embeddings")
    rerank_model_name: str = Field(default="qwen3-reranker-8b", description="The model name to use for reranking")
    task: Literal["retrieval", "clustering"] = Field(default="retrieval", description="The embedding task type")
    batch_size: int = Field(default=8, description="Batch size for processing multiple texts")
    show_progress: bool = Field(default=False, description="Whether to show progress bar for batch processing")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    @computed_field
    @property
    def embeddings_endpoint(self) -> str:
        """Compute the embeddings endpoint URL."""
        return f"{self.api_url.rstrip('/')}/v1/embeddings"
    
    @computed_field
    @property
    def rerank_endpoint(self) -> str:
        """Compute the rerank endpoint URL."""
        return f"{self.api_url.rstrip('/')}/v1/rerank"

    def _get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Sends a batch of texts to the Qwen embeddings endpoint.

        Args:
            texts: A list of strings to embed.
            is_query: Whether these are query texts (for retrieval task).

        Returns:
            A list of embedding vectors (list of floats), corresponding to the input texts.

        Raises:
            requests.exceptions.RequestException: If the API call fails.
            ValueError: If the API response is unexpected.
        """
        if not texts:
            return []

        # For retrieval task with single query, the API will handle instruction
        # For multiple texts in retrieval, we let the API handle it
        payload = {
            "model": self.model_name,
            "input": texts,
            "task": self.task
        }

        try:
            response = requests.post(
                self.embeddings_endpoint, 
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if 'data' not in data or not isinstance(data['data'], list):
                raise ValueError("Unexpected API response format: missing 'data' list")
            
            if len(data['data']) != len(texts):
                print(f"Warning: Number of embeddings returned ({len(data['data'])}) does not match number of inputs ({len(texts)})")
                if all('embedding' in item for item in data['data']):
                    return [item['embedding'] for item in data['data']]
                else:
                    raise ValueError("Unexpected API response format: items in 'data' missing 'embedding'")

            # Extract and return embeddings
            embeddings = [item['embedding'] for item in data['data']]
            return embeddings

        except requests.exceptions.RequestException as e:
            print(f"API call failed for batch: {e}")
            if 'response' in locals() and response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute doc embeddings using the Qwen server.
        
        This is the main LangChain interface method for embedding documents.
        For retrieval task, documents don't get instruction prefixes.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []
            
        # Clean texts by replacing newlines with spaces
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        all_embeddings = []
        num_texts = len(cleaned_texts)

        if num_texts <= self.batch_size:
            # Process all at once if within batch size
            return self._get_embeddings_batch(cleaned_texts, is_query=False)
        
        # Process in batches
        if self.show_progress:
            print(f"Processing {num_texts} texts in batches of {self.batch_size}...")
            iterator = tqdm(
                range(0, num_texts, self.batch_size), 
                desc="Getting embeddings from Qwen"
            )
        else:
            iterator = range(0, num_texts, self.batch_size)

        for i in iterator:
            batch_texts = cleaned_texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._get_embeddings_batch(batch_texts, is_query=False)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"\nError processing batch starting at index {i}. Stopping.")
                raise e

        # Verify we got embeddings for all texts
        if len(all_embeddings) != num_texts:
            print(f"Warning: Expected {num_texts} embeddings but received {len(all_embeddings)}.")

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Compute query embeddings using the Qwen server.
        
        This is the main LangChain interface method for embedding queries.
        For retrieval task, queries get instruction prefixes.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        # For a single query in retrieval task, the API will add instruction
        return self._get_embeddings_batch([text], is_query=True)[0]

    def rerank(
        self, 
        query: str, 
        documents: List[Union[str, Dict[str, Any]]], 
        task: Optional[str] = None,
        top_n: Optional[int] = None,
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The query text.
            documents: List of documents (strings or dicts with 'text' field).
            task: Optional task description (defaults to web search retrieval).
            top_n: Number of top results to return.
            return_documents: Whether to return document texts in results.

        Returns:
            List of reranked results with scores and documents.
        """
        if not documents:
            return []

        payload = {
            "model": self.rerank_model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents
        }
        
        if task:
            payload["task"] = task

        try:
            response = requests.post(
                self.rerank_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            
            if 'results' not in data:
                raise ValueError("Unexpected API response format: missing 'results'")
            
            return data['results']

        except requests.exceptions.RequestException as e:
            print(f"Reranking API call failed: {e}")
            if 'response' in locals() and response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
            raise

    def compute_similarity(self, embeddings1: List[List[float]], embeddings2: List[List[float]]) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
            
        Returns:
            Similarity matrix of shape (len(embeddings1), len(embeddings2)).
        """
        # Convert to numpy arrays
        emb1 = np.array(embeddings1)
        emb2 = np.array(embeddings2)
        
        # Embeddings from the API are already normalized, so dot product = cosine similarity
        return np.dot(emb1, emb2.T)

    def embed_dataframe_column(self, df: pd.DataFrame, column_name: str) -> List[List[float]]:
        """
        Gets embeddings for all texts in a DataFrame column.
        
        This method is kept for backward compatibility with existing code.

        Args:
            df: The pandas DataFrame.
            column_name: The name of the column containing texts to embed.

        Returns:
            A list of embedding vectors, in the same order as the DataFrame rows.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        texts = df[column_name].tolist()
        return self.embed_documents(texts)

    def rerank_dataframe(
        self, 
        query: str, 
        df: pd.DataFrame, 
        text_column: str,
        top_n: Optional[int] = None,
        score_column: str = "rerank_score"
    ) -> pd.DataFrame:
        """
        Rerank documents in a DataFrame based on query relevance.
        
        Args:
            query: The query text.
            df: DataFrame containing documents.
            text_column: Column name containing document texts.
            top_n: Number of top results to return.
            score_column: Name for the new score column.
            
        Returns:
            DataFrame sorted by relevance with added score column.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        
        # Get documents from dataframe
        documents = df[text_column].tolist()
        
        # Rerank
        results = self.rerank(query, documents, top_n=top_n, return_documents=False)
        
        # Create a mapping of original index to score
        score_map = {result['index']: result['relevance_score'] for result in results}
        
        # Add scores to dataframe
        df_copy = df.copy()
        df_copy[score_column] = [score_map.get(i, 0.0) for i in range(len(df))]
        
        # Sort by score and filter top_n if specified
        df_sorted = df_copy.sort_values(score_column, ascending=False)
        
        if top_n:
            df_sorted = df_sorted.head(top_n)
        
        return df_sorted

    def get_available_models(self) -> List[str]:
        """Get list of available models from the API."""
        try:
            response = requests.get(f"{self.api_url.rstrip('/')}/v1/models", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        except Exception as e:
            print(f"Failed to get models: {e}")
            return []

    def get_available_tasks(self) -> Dict[str, str]:
        """Get available embedding tasks from the API."""
        try:
            response = requests.get(f"{self.api_url.rstrip('/')}/v1/tasks", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('embedding_tasks', {})
        except Exception as e:
            print(f"Failed to get tasks: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(f"{self.api_url.rstrip('/')}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}