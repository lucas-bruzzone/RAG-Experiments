"""Embeddings Manager - Singleton for embedding model"""

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingsManager:
    """Singleton manager for embedding model"""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model_name = model_name
            cls._model = None
        return cls._instance
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model"""
        if self._model is None:
            print(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Encode texts to embeddings"""
        return self.model.encode(texts, **kwargs)
    
    def encode_query(self, query: str) -> List[float]:
        """Encode single query"""
        return self.model.encode([query])[0].tolist()
