import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, computed_field


class InfinityEmbeddings(BaseModel, Embeddings):
    """
    A LangChain-compatible class to get embeddings from a running infinity server API.
    
    Example:
        .. code-block:: python

            from util.infinityEmbedding import InfinityEmbeddings

            embeddings = InfinityEmbeddings(
                api_url="http://172.16.32.24:8005",
                model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
            )
            
            # Use with LangChain
            docs = ["Hello world", "Another document"]
            doc_embeddings = embeddings.embed_documents(docs)
            
            query = "What is this about?"
            query_embedding = embeddings.embed_query(query)
    """
    
    api_url: str = Field(..., description="The base URL of the infinity server")
    model_name: str = Field(..., description="The model name to use for embeddings")
    batch_size: int = Field(default=32, description="Batch size for processing multiple texts")
    show_progress: bool = Field(default=False, description="Whether to show progress bar for batch processing")
    
    @computed_field
    @property
    def embeddings_endpoint(self) -> str:
        """Compute the embeddings endpoint URL."""
        return f"{self.api_url.rstrip('/')}/embeddings"

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Sends a batch of texts to the infinity embeddings endpoint.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors (list of floats), corresponding to the input texts.

        Raises:
            requests.exceptions.RequestException: If the API call fails.
            ValueError: If the API response is unexpected.
        """
        if not texts:
            return []

        payload = {
            "model": self.model_name,
            "input": texts
        }

        try:
            response = requests.post(self.embeddings_endpoint, json=payload)
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
        Compute doc embeddings using the Infinity server.
        
        This is the main LangChain interface method for embedding documents.

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
            return self._get_embeddings_batch(cleaned_texts)
        
        # Process in batches
        if self.show_progress:
            print(f"Processing {num_texts} texts in batches of {self.batch_size}...")
            iterator = tqdm(
                range(0, num_texts, self.batch_size), 
                desc="Getting embeddings from Infinity"
            )
        else:
            iterator = range(0, num_texts, self.batch_size)

        for i in iterator:
            batch_texts = cleaned_texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._get_embeddings_batch(batch_texts)
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
        Compute query embeddings using the Infinity server.
        
        This is the main LangChain interface method for embedding queries.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    # Keep the original DataFrame method for backward compatibility
    def embed_dataframe_column(self, df: pd.DataFrame, column_name: str) -> List[List[float]]:
        """
        Gets embeddings for all texts in a DataFrame column.
        
        This method is kept for backward compatibility with your existing code.

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