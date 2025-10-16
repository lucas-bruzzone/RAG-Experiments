"""Document Processing Pipeline"""

from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

from ..core.embeddings import EmbeddingsManager
from ..core.vectorstore import VectorStore


class DocumentProcessor:
    """Process and index documents into vector store"""
    
    def __init__(self, config: dict):
        self.config = config
        
        doc_config = config.get("document", {})
        self.chunk_size = doc_config.get("chunk_size", 1000)
        self.chunk_overlap = doc_config.get("chunk_overlap", 200)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        vectorstore_config = config.get("vectorstore", {})
        self.vectorstore = VectorStore(
            persist_directory=vectorstore_config.get("persist_directory", "./db/chroma"),
            collection_name=vectorstore_config.get("collection_name", "documents")
        )
        
        embeddings_config = config.get("embeddings", {})
        self.embeddings = EmbeddingsManager(
            model_name=embeddings_config.get("model", "all-MiniLM-L6-v2")
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on extension"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def index_documents(self, chunks: List[Document], source_file: str):
        """Index chunks into vector store"""
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {**chunk.metadata, "source_file": source_file} 
            for chunk in chunks
        ]
        
        # Generate embeddings
        embeddings = self.embeddings.encode(texts, show_progress_bar=True)
        
        # Generate IDs
        base_id = Path(source_file).stem
        ids = [f"{base_id}_{i}" for i in range(len(texts))]
        
        # Add to vectorstore
        self.vectorstore.add(
            texts=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
    
    def process_file(self, file_path: str) -> int:
        """
        Complete pipeline: load -> split -> index
        Returns number of chunks indexed
        """
        print(f"\nProcessing: {file_path}")
        
        documents = self.load_document(file_path)
        print(f"  Loaded: {len(documents)} pages")
        
        chunks = self.split_documents(documents)
        print(f"  Split into: {len(chunks)} chunks")
        
        self.index_documents(chunks, file_path)
        print(f"  Indexed: {len(chunks)} chunks")
        
        return len(chunks)
    
    def process_directory(self, directory: str) -> int:
        """Process all supported files in directory"""
        data_dir = Path(directory)
        
        supported = self.config.get("document", {}).get("supported_formats", [".pdf", ".txt"])
        files = []
        for ext in supported:
            files.extend(data_dir.glob(f"*{ext}"))
        
        if not files:
            print(f"No files found in {directory}")
            return 0
        
        print(f"\nFound {len(files)} files")
        total_chunks = 0
        
        for file in files:
            try:
                chunks = self.process_file(str(file))
                total_chunks += chunks
            except Exception as e:
                print(f"  Error processing {file.name}: {e}")
        
        return total_chunks
    
    def get_stats(self):
        """Get collection statistics"""
        return self.vectorstore.get_stats()
