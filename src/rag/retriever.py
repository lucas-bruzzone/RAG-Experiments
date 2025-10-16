"""Retrieval logic for RAG"""

from typing import List, Dict
from ..core.embeddings import EmbeddingsManager
from ..core.vectorstore import VectorStore


class Retriever:
    """Handles document retrieval from vector store"""
    
    def __init__(self, config: dict):
        self.config = config
        
        retrieval_config = config.get("retrieval", {})
        self.top_k = retrieval_config.get("top_k", 3)
        self.score_threshold = retrieval_config.get("score_threshold", 0.0)
        
        vectorstore_config = config.get("vectorstore", {})
        self.vectorstore = VectorStore(
            persist_directory=vectorstore_config.get("persist_directory", "./db/chroma"),
            collection_name=vectorstore_config.get("collection_name", "documents")
        )
        
        embeddings_config = config.get("embeddings", {})
        self.embeddings = EmbeddingsManager(
            model_name=embeddings_config.get("model", "all-MiniLM-L6-v2")
        )
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant documents for query
        
        Returns list of dicts with: content, metadata, rank
        """
        k = top_k if top_k is not None else self.top_k
        
        # Generate query embedding
        query_embedding = self.embeddings.encode_query(query)
        
        # Search vector store
        results = self.vectorstore.query(query_embedding, top_k=k)
        
        # Format results
        retrieved_docs = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            retrieved_docs.append({
                "content": doc,
                "metadata": metadata,
                "rank": i + 1
            })
        
        return retrieved_docs
