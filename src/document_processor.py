"""Document processing and ingestion pipeline"""

# Fix SQLite para ChromaDB
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from pathlib import Path
from typing import List
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer


class DocumentProcessor:
    """Processa e indexa documentos no vector store"""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Inicializa ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Cria ou obtém collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Collection '{collection_name}' carregada")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"Collection '{collection_name}' criada")

        # Inicializa modelo de embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Carrega documento baseado na extensão"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path))
        else:
            raise ValueError(f"Formato não suportado: {path.suffix}")

        documents = loader.load()
        print(f"✓ Carregado: {path.name} ({len(documents)} páginas/docs)")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos em chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Dividido em {len(chunks)} chunks")
        return chunks

    def index_documents(self, chunks: List[Document]):
        """Indexa chunks no vector store"""
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Gera embeddings
        print("Gerando embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Adiciona ao ChromaDB
        ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"✓ {len(chunks)} chunks indexados")

    def process_file(self, file_path: str):
        """Pipeline completo: carrega -> divide -> indexa"""
        print(f"\n{'='*50}")
        print(f"Processando: {file_path}")
        print("=" * 50)

        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        self.index_documents(chunks)

        print(f"✓ Processamento completo!")

    def get_collection_stats(self):
        """Retorna estatísticas da collection"""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    # Exemplo de uso
    processor = DocumentProcessor()

    # Cria diretório de dados se não existir
    os.makedirs("data", exist_ok=True)

    print("\nProcessador de documentos iniciado!")
    print("\nUso:")
    print("  processor = DocumentProcessor()")
    print("  processor.process_file('data/seu_arquivo.pdf')")
    print("\nEstatísticas:")
    print(processor.get_collection_stats())
