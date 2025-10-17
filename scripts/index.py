#!/usr/bin/env python3
"""Index documents into vector store"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.document import DocumentProcessor
from src.utils.config import load_config
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


def create_sample_document():
    """Create sample document for testing"""
    sample_content = """Introduction to RAG (Retrieval Augmented Generation)

RAG is a technique that combines information retrieval with language generation.
The process involves three main stages:

1. Indexing: Documents are split into chunks and converted to embeddings
2. Retrieval: Relevant chunks are retrieved based on similarity
3. Generation: An LLM generates responses using retrieved chunks as context

Advantages of RAG:
- Reduces model hallucinations
- Enables domain-specific knowledge
- Does not require fine-tuning the base model
- Knowledge can be easily updated

Vector Databases:
ChromaDB, Pinecone, Weaviate and FAISS are examples of vector databases
used to store and retrieve embeddings efficiently.

Embeddings:
Dense vector representations of text that capture semantic meaning.
Models like sentence-transformers generate high-quality embeddings."""

    sample_file = Path("data/raw/sample_document.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text(sample_content, encoding="utf-8")
    
    logger.info(f"Created sample document: {sample_file}")
    print(f"Created sample document: {sample_file}")
    return str(sample_file)


def main():
    parser = argparse.ArgumentParser(description="Index documents into vector store")
    parser.add_argument("--file", help="Index specific file")
    parser.add_argument("--dir", help="Index all files in directory")
    parser.add_argument("--create-sample", action="store_true", 
                       help="Create and index sample document")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset collection before indexing")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, verbose=args.verbose)
    logger.info("Starting indexing script")
    
    if not any([args.file, args.dir, args.create_sample]):
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    processor = DocumentProcessor(config)
    
    # Reset if requested
    if args.reset:
        logger.info("Resetting collection")
        print("Resetting collection...")
        processor.vectorstore.reset()
    
    # Index documents
    total_chunks = 0
    
    if args.create_sample:
        sample_file = create_sample_document()
        total_chunks += processor.process_file(sample_file)
    
    if args.file:
        total_chunks += processor.process_file(args.file)
    
    if args.dir:
        total_chunks += processor.process_directory(args.dir)
    
    # Show statistics
    print("\n" + "=" * 60)
    print("Indexing complete!")
    stats = processor.get_stats()
    print(f"Collection: {stats['name']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Database: {stats['persist_directory']}")
    print("=" * 60)
    
    logger.info(
        "Indexing completed",
        extra={
            'total_chunks': total_chunks,
            'collection': stats['name']
        }
    )


if __name__ == "__main__":
    main()