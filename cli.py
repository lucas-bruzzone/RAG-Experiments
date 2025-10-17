#!/usr/bin/env python3
"""Unified CLI for RAG operations"""

import argparse
import sys
from pathlib import Path

from src.processing.document import DocumentProcessor
from src.rag.chain import RAGChain
from src.rag.retriever import Retriever
from src.utils.config import load_config
from src.telemetry import PhoenixTelemetry


def cmd_index(args):
    """Index documents"""
    config = load_config()
    processor = DocumentProcessor(config)

    if args.reset:
        processor.vectorstore.reset()
        print("Collection reset")

    if args.file:
        processor.process_file(args.file)
    elif args.dir:
        processor.process_directory(args.dir)
    else:
        print("Specify --file or --dir")
        return

    print("\nStatistics:")
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def cmd_search(args):
    """Semantic search (retrieval only)"""
    if args.telemetry:
        PhoenixTelemetry.start()

    config = load_config()
    retriever = Retriever(config)

    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Top {args.top_k} results:\n")

    for doc in results:
        print(f"[{doc['rank']}] {doc['content'][:200]}...\n")


def cmd_query(args):
    """Complete RAG query (retrieval + generation)"""
    if args.telemetry:
        PhoenixTelemetry.start()

    config = load_config()
    rag = RAGChain(config, llm_provider=args.llm)

    result = rag.query(
        args.query, return_context=args.show_context, top_k=args.top_k)

    print(f"\nANSWER:\n{result['answer']}")

    if args.show_context:
        print(f"\nCONTEXT ({result['context_used']} docs):")
        for doc in result["context_docs"]:
            print(f"[{doc['rank']}] {doc['content'][:200]}...\n")


def cmd_chat(args):
    """Interactive chat mode"""
    if args.telemetry:
        PhoenixTelemetry.start()

    config = load_config()
    rag = RAGChain(config, llm_provider=args.llm)
    rag.chat()


def cmd_stats(args):
    """Show statistics"""
    config = load_config()
    processor = DocumentProcessor(config)
    stats = processor.get_stats()

    print("\nCollection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument("--telemetry", action="store_true",
                        help="Enable Phoenix telemetry UI")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--file", help="Specific file")
    index_parser.add_argument("--dir", help="Directory of files")
    index_parser.add_argument(
        "--reset", action="store_true", help="Reset collection")

    # Search
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", help="Search text")
    search_parser.add_argument("--top-k", type=int, default=3)

    # Query
    query_parser = subparsers.add_parser("query", help="Complete RAG query")
    query_parser.add_argument("query", help="Question")
    query_parser.add_argument("--top-k", type=int, default=3)
    query_parser.add_argument("--llm", default="auto",
                              choices=["auto", "ollama", "openai"])
    query_parser.add_argument("--show-context", action="store_true")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Interactive mode")
    chat_parser.add_argument("--llm", default="auto",
                             choices=["auto", "ollama", "openai"])

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

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

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if hasattr(args, 'telemetry') and args.telemetry:
            PhoenixTelemetry.stop()


if __name__ == "__main__":
    main()
