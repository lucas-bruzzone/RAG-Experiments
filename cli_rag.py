#!/usr/bin/env python3
"""CLI para RAG - Interface de linha de comando"""

# Fix SQLite
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import argparse
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.rag_chain import RAGChain


def cmd_index(args):
    """Indexa documentos"""
    processor = DocumentProcessor(
        collection_name=args.collection,
        persist_directory=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.file:
        processor.process_file(args.file)
    elif args.dir:
        data_dir = Path(args.dir)
        files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
        print(f"Encontrados {len(files)} arquivos")
        for file in files:
            processor.process_file(str(file))

    print("\nEstatísticas:")
    print(processor.get_collection_stats())


def cmd_search(args):
    """Busca semântica (retrieval only)"""
    from sentence_transformers import SentenceTransformer
    import chromadb

    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(args.collection)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_emb = model.encode([args.query])[0]
    results = collection.query(
        query_embeddings=[query_emb.tolist()], n_results=args.top_k
    )

    print(f"\nQuery: {args.query}")
    print(f"Top {args.top_k} resultados:\n")
    for i, doc in enumerate(results["documents"][0], 1):
        print(f"[{i}] {doc}\n")


def cmd_query(args):
    """Query RAG completa (retrieval + generation)"""
    rag = RAGChain(
        collection_name=args.collection,
        persist_directory=args.db_path,
        top_k=args.top_k,
        llm_provider=args.llm,
    )

    result = rag.query(args.query, return_context=args.show_context)

    print(f"\nRESPOSTA:\n{result['answer']}")

    if args.show_context:
        print(f"\nCONTEXTO ({result['context_used']} docs):")
        for doc in result["context_docs"]:
            print(f"[{doc['rank']}] {doc['content'][:200]}...\n")


def cmd_chat(args):
    """Modo chat interativo"""
    rag = RAGChain(
        collection_name=args.collection,
        persist_directory=args.db_path,
        top_k=args.top_k,
        llm_provider=args.llm,
    )
    rag.chat()


def cmd_stats(args):
    """Mostra estatísticas"""
    import chromadb

    client = chromadb.PersistentClient(path=args.db_path)

    try:
        collection = client.get_collection(args.collection)
        count = collection.count()
        print(f"\nCollection: {args.collection}")
        print(f"Total chunks: {count}")
        print(f"Database: {args.db_path}")
    except:
        print(f"Collection '{args.collection}' não encontrada")


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument("--collection", default="documents", help="Nome da collection")
    parser.add_argument("--db-path", default="./chroma_db", help="Path do ChromaDB")

    subparsers = parser.add_subparsers(dest="command", help="Comandos")

    # Index
    index_parser = subparsers.add_parser("index", help="Indexar documentos")
    index_parser.add_argument("--file", help="Arquivo específico")
    index_parser.add_argument("--dir", help="Diretório de arquivos")
    index_parser.add_argument("--chunk-size", type=int, default=1000)
    index_parser.add_argument("--chunk-overlap", type=int, default=200)

    # Search
    search_parser = subparsers.add_parser("search", help="Busca semântica")
    search_parser.add_argument("query", help="Texto da busca")
    search_parser.add_argument("--top-k", type=int, default=3)

    # Query
    query_parser = subparsers.add_parser("query", help="Query RAG completa")
    query_parser.add_argument("query", help="Pergunta")
    query_parser.add_argument("--top-k", type=int, default=3)
    query_parser.add_argument("--llm", default="ollama", choices=["ollama", "openai"])
    query_parser.add_argument("--show-context", action="store_true")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Modo interativo")
    chat_parser.add_argument("--top-k", type=int, default=3)
    chat_parser.add_argument("--llm", default="ollama", choices=["ollama", "openai"])

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Estatísticas")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "query": cmd_query,
        "chat": cmd_chat,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
