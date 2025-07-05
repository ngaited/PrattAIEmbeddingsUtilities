import requests
from typing import List, Dict, Any, Sequence, Optional

from langchain_core.documents import Document
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field, PrivateAttr


class InfinityReranker(BaseDocumentCompressor):
    """
    A LangChain-compatible document compressor for reranking tasks using a running
    infinity server API.
    """

    api_url: str = Field(..., description="The base URL of the infinity server")
    model_name: str = Field(..., description="The reranker model name to use")
    top_n: int = Field(
        default=3,
        description="The number of top-ranking documents to return.",
        ge=1
    )
    
    _session: Any = PrivateAttr()

    def __init__(self, **data: Any):
        """Initialize the reranker and a requests session."""
        super().__init__(**data)
        self._session = requests.Session()

    @property
    def rerank_endpoint(self) -> str:
        """The computed rerank endpoint URL."""
        return f"{self.api_url.rstrip('/')}/rerank"

    def _call_api(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """Sends a query and documents to the infinity rerank endpoint."""
        if not documents:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }

        try:
            response = self._session.post(self.rerank_endpoint, json=payload, timeout=300)
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
        results = self._call_api(query=query, documents=documents)
        
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
        results = self._call_api(query=query, documents=doc_texts)

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