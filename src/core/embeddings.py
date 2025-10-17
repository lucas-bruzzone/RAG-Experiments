"""Embeddings Manager - Singleton for embedding model"""

from typing import List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Singleton manager for embedding model"""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model_name = model_name
            cls._model = None
            logger.debug(f"EmbeddingsManager instance created for model: {model_name}")
        return cls._instance
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            try:
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Embedding model loaded successfully: {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                raise
        return self._model
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Encode texts to embeddings"""
        logger.debug(f"Encoding {len(texts)} texts to embeddings")
        try:
            embeddings = self.model.encode(texts, **kwargs)
            logger.debug(f"Successfully encoded {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}", exc_info=True)
            raise
    
    def encode_query(self, query: str) -> List[float]:
        """Encode single query"""
        logger.debug(f"Encoding query: '{query[:50]}...'")
        return self.model.encode([query])[0].tolist()