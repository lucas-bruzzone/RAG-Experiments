"""Document Processing Pipeline"""

from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

from ..core.embeddings import EmbeddingsManager
from ..core.vectorstore import VectorStore
from ..utils import get_logger, timed

logger = get_logger(__name__)


class DocumentProcessor:
    """Process and index documents into vector store"""
    
    def __init__(self, config: dict):
        self.config = config
        
        doc_config = config.get("document", {})
        self.chunk_size = doc_config.get("chunk_size", 1000)
        self.chunk_overlap = doc_config.get("chunk_overlap", 200)
        
        logger.info(
            "Initializing DocumentProcessor",
            extra={
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        )
        
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
    
    @timed
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on extension"""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.debug(f"Loading document: {file_path}")
        
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path))
        else:
            logger.error(f"Unsupported format: {path.suffix}")
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from {path.name}")
        return docs
    
    @timed
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.debug(f"Split into {len(chunks)} chunks")
        return chunks
    
    @timed
    def index_documents(self, chunks: List[Document], source_file: str):
        """Index chunks into vector store"""
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {**chunk.metadata, "source_file": source_file} 
            for chunk in chunks
        ]
        
        logger.debug(f"Generating embeddings for {len(texts)} chunks")
        
        # Generate embeddings
        embeddings = self.embeddings.encode(texts, show_progress_bar=True)
        
        # Generate IDs
        base_id = Path(source_file).stem
        ids = [f"{base_id}_{i}" for i in range(len(texts))]
        
        logger.debug(f"Adding {len(texts)} chunks to vector store")
        
        # Add to vectorstore
        self.vectorstore.add(
            texts=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(
            f"Indexed {len(chunks)} chunks from {Path(source_file).name}",
            extra={'chunks': len(chunks), 'source': source_file}
        )
    
    @timed
    def process_file(self, file_path: str) -> int:
        """
        Complete pipeline: load -> split -> index
        Returns number of chunks indexed
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            documents = self.load_document(file_path)
            chunks = self.split_documents(documents)
            self.index_documents(chunks, file_path)
            
            logger.info(
                f"Successfully processed {file_path}",
                extra={
                    'file': file_path,
                    'pages': len(documents),
                    'chunks': len(chunks)
                }
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error(
                f"Failed to process {file_path}: {e}",
                extra={'file': file_path},
                exc_info=True
            )
            raise
    
    def process_directory(self, directory: str) -> int:
        """Process all supported files in directory"""
        data_dir = Path(directory)
        
        supported = self.config.get("document", {}).get("supported_formats", [".pdf", ".txt"])
        files = []
        for ext in supported:
            files.extend(data_dir.glob(f"*{ext}"))
        
        if not files:
            logger.warning(f"No files found in {directory}")
            return 0
        
        logger.info(f"Found {len(files)} files in {directory}")
        total_chunks = 0
        failed_files = []
        
        for file in files:
            try:
                chunks = self.process_file(str(file))
                total_chunks += chunks
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                failed_files.append(file.name)
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
        
        logger.info(
            f"Directory processing complete",
            extra={
                'total_files': len(files),
                'successful': len(files) - len(failed_files),
                'failed': len(failed_files),
                'total_chunks': total_chunks
            }
        )
        
        return total_chunks
    
    def get_stats(self):
        """Get collection statistics"""
        stats = self.vectorstore.get_stats()
        logger.debug(f"Collection stats: {stats}")
        return stats