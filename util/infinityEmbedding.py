import requests
from typing import List, Dict, Any, Sequence, Optional, Union, Literal
import numpy as np
import pandas as pd
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, PrivateAttr, computed_field


class InfinityEmbeddingsReranker(BaseDocumentCompressor, BaseModel, Embeddings):
    """
    A LangChain-compatible document compressor and embeddings class for both reranking 
    and embedding tasks using a running infinity server API.
    
    This class can serve as a drop-in replacement for QwenEmbeddings while also 
    providing reranking functionality.
    
    Example:
        .. code-block:: python

            from util.infinityEmbedding import InfinityEmbeddingsReranker

            # For embeddings only
            embeddings = InfinityEmbeddingsReranker(
                api_url="http://localhost:8006",
                model_name="BAAI/bge-base-en-v1.5"
            )
            
            # For both embeddings and reranking
            reranker = InfinityEmbeddingsReranker(
                api_url="http://localhost:8006",
                model_name="BAAI/bge-base-en-v1.5",
                rerank_model_name="Alibaba-NLP/gte-multilingual-reranker-base"
            )
            
            # Use with LangChain
            docs = ["Hello world", "Another document"]
            doc_embeddings = embeddings.embed_documents(docs)
            
            query = "What is this about?"
            query_embedding = embeddings.embed_query(query)
            
            # Use for reranking
            reranked_docs = reranker.compress_documents(docs, query)
            
            # Use batch processing for efficiency
            batch_result = embeddings.embed_batch(
                queries=["query1", "query2"],
                documents=["doc1", "doc2", "doc3"]
            )
    
    Available Models on Backend:
        Embedding Models:
        - BAAI/bge-base-en-v1.5 (English, 768 dimensions)
        - nomic-ai/nomic-embed-text-v1.5 (Multilingual, 768 dimensions)
        - nomic-ai/nomic-embed-text-v2-moe (Multilingual, variable dimensions)
        
        Reranking Models:
        - Alibaba-NLP/gte-multilingual-reranker-base (Multilingual)
    """

    api_url: str = Field(..., description="The base URL of the infinity server")
    model_name: str = Field(..., description="The model name to use for embeddings/reranking. Available models: BAAI/bge-base-en-v1.5, nomic-ai/nomic-embed-text-v1.5, nomic-ai/nomic-embed-text-v2-moe")
    rerank_model_name: Optional[str] = Field(
        default=None, 
        description="Optional separate model name for reranking. If None, uses model_name. Available reranking models: Alibaba-NLP/gte-multilingual-reranker-base"
    )
    top_n: int = Field(
        default=3,
        description="The number of top-ranking documents to return for reranking.",
        ge=1
    )
    task: Literal["retrieval", "clustering"] = Field(
        default="retrieval", 
        description="The embedding task type"
    )
    dimensions: Optional[int] = Field(
        default=None, 
        description="Number of dimensions for Matryoshka embeddings"
    )
    batch_size: int = Field(
        default=32, 
        description="Batch size for processing multiple texts"
    )
    show_progress: bool = Field(
        default=False, 
        description="Whether to show progress bar for batch processing"
    )
    timeout: int = Field(
        default=300, 
        description="Request timeout in seconds"
    )
    
    _session: Any = PrivateAttr()

    def __init__(self, **data: Any):
        """Initialize the embeddings/reranker and a requests session."""
        super().__init__(**data)
        self._session = requests.Session()

    @computed_field
    @property
    def embeddings_endpoint(self) -> str:
        """Compute the embeddings endpoint URL."""
        return f"{self.api_url.rstrip('/')}/embeddings"
    
    @computed_field
    @property
    def batch_endpoint(self) -> str:
        """Compute the batch embeddings endpoint URL."""
        return f"{self.api_url.rstrip('/')}/embeddings/batch"

    @property
    def rerank_endpoint(self) -> str:
        """The computed rerank endpoint URL."""
        return f"{self.api_url.rstrip('/')}/rerank"

    # ========== EMBEDDINGS FUNCTIONALITY ==========

    def _get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Sends a batch of texts to the infinity embeddings endpoint.

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

        # Build payload with Matryoshka dimension support
        payload = {
            "model": self.model_name,
            "input": texts,
            "task": self.task
        }
        
        # Add dimensions if specified (Matryoshka support)
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        try:
            response = self._session.post(
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
        Compute doc embeddings using the Infinity server.
        
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
                desc="Getting embeddings from Infinity"
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
        Compute query embeddings using the Infinity server.
        
        This is the main LangChain interface method for embedding queries.
        For retrieval task, queries get instruction prefixes.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        # For a single query in retrieval task, the API will add instruction
        return self._get_embeddings_batch([text], is_query=True)[0]

    def embed_batch(
        self,
        queries: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Efficiently process queries and documents separately using the batch endpoint.
        
        Args:
            queries: List of query texts.
            documents: List of document texts.
            dimensions: Override dimensions for this specific call.
            
        Returns:
            Dictionary containing query_embeddings, document_embeddings, and optionally similarity_scores.
        """
        if not queries and not documents:
            raise ValueError("Either queries or documents must be provided")

        payload = {
            "task": self.task
        }
        
        if queries:
            payload["queries"] = queries
        if documents:
            payload["documents"] = documents
            
        # Use instance dimensions or override
        dims = dimensions if dimensions is not None else self.dimensions
        if dims is not None:
            payload["dimensions"] = dims

        try:
            response = self._session.post(
                self.batch_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Batch API call failed: {e}")
            if 'response' in locals() and response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
            raise

    # ========== RERANKING FUNCTIONALITY ==========

    def _call_rerank_api(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """Sends a query and documents to the infinity rerank endpoint."""
        if not documents:
            return []

        # Use rerank_model_name if specified, otherwise use model_name
        model_to_use = self.rerank_model_name or self.model_name

        payload = {
            "model": model_to_use,
            "query": query,
            "documents": documents,
        }

        try:
            response = self._session.post(self.rerank_endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if "results" not in data or not isinstance(data["results"], list):
                raise ValueError("Unexpected API response format: missing 'results' list")

            # Add the original document text to each result
            results_with_docs = []
            for result in data["results"]:
                if "index" in result:
                    result["document"] = documents[result["index"]]
                results_with_docs.append(result)
            
            return results_with_docs

        except requests.exceptions.RequestException as e:
            print(f"API call to {self.rerank_endpoint} failed: {e}")
            if 'response' in locals() and response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
            raise

    def rank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """A direct method to get reranking scores for a list of document strings."""
        results = self._call_rerank_api(query=query, documents=documents)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return results

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents by reranking them relative to the query."""
        if not documents:
            return []

        doc_texts = [doc.page_content for doc in documents]
        results = self._call_rerank_api(query=query, documents=doc_texts)

        final_docs = []
        for result in results:
            original_index = result.get("index")
            relevance_score = result.get("relevance_score")

            if original_index is None or relevance_score is None:
                continue
            
            doc = documents[original_index].copy()
            doc.metadata["relevance_score"] = relevance_score
            final_docs.append(doc)

        return final_docs[:self.top_n]

    # ========== ADDITIONAL UTILITY METHODS ==========

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

    def search_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        return_scores: bool = True
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Find most similar documents to a query using embeddings.
        
        Args:
            query: The query text.
            documents: List of document texts to search.
            top_k: Number of top results to return.
            return_scores: Whether to return similarity scores.
            
        Returns:
            List of documents or list of dicts with document and score.
        """
        if not documents:
            return []

        # Use batch endpoint for efficiency
        batch_result = self.embed_batch(queries=[query], documents=documents)
        
        if 'similarity_scores' in batch_result:
            # API computed similarities for us
            scores = batch_result['similarity_scores'][0]  # First (and only) query
        else:
            # Compute similarities manually
            query_emb = batch_result['query_embeddings'][0]
            doc_embs = batch_result['document_embeddings']
            scores = np.dot(doc_embs, query_emb).tolist()
        
        # Create results with scores and indices
        results = [(i, score, doc) for i, (score, doc) in enumerate(zip(scores, documents))]
        
        # Sort by score (descending) and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        if return_scores:
            return [{"document": doc, "score": score, "index": idx} for idx, score, doc in results]
        else:
            return [doc for _, _, doc in results]

    # ========== API INFORMATION METHODS ==========

    def get_available_models(self) -> List[str]:
        """Get list of available models from the API."""
        try:
            response = self._session.get(f"{self.api_url.rstrip('/')}/models", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        except Exception as e:
            print(f"Failed to get models: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = self._session.get(f"{self.api_url.rstrip('/')}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def set_dimensions(self, dimensions: Optional[int]) -> None:
        """
        Set the dimensions for Matryoshka embeddings.
        
        Args:
            dimensions: Number of dimensions, or None for full dimensions.
        """
        self.dimensions = dimensions

    def with_dimensions(self, dimensions: Optional[int]) -> 'InfinityEmbeddingsReranker':
        """
        Create a new instance with different dimensions.
        
        Args:
            dimensions: Number of dimensions for the new instance.
            
        Returns:
            New InfinityEmbeddingsReranker instance with specified dimensions.
        """
        return self.model_copy(update={"dimensions": dimensions})