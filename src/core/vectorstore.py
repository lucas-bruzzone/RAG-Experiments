"""Vector Store Wrapper - ChromaDB abstraction"""

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from typing import List, Dict
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Wrapper for ChromaDB operations"""
    
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
    
    def add(self, texts: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict], ids: List[str]):
        """Add documents to collection"""
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        """Query similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    
    def count(self) -> int:
        """Count documents in collection"""
        return self.collection.count()
    
    def reset(self):
        """Delete and recreate collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "name": self.collection_name,
            "total_chunks": self.count(),
            "persist_directory": self.persist_directory
        }
