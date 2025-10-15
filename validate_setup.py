#!/usr/bin/env python3
"""Validação da configuração do ambiente RAG - Etapa 2"""

# Fix SQLite para ChromaDB
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def validate_imports():
    """Testa importação das bibliotecas principais"""
    try:
        import langchain

        print("✓ LangChain instalado")

        import chromadb

        print("✓ ChromaDB instalado")

        from sentence_transformers import SentenceTransformer

        print("✓ Sentence Transformers instalado")

        import pypdf

        print("✓ PyPDF instalado")

        from dotenv import load_dotenv

        print("✓ python-dotenv instalado")

        return True
    except ImportError as e:
        print(f"✗ Erro de importação: {e}")
        return False


def test_chromadb():
    """Testa ChromaDB básico"""
    try:
        import chromadb

        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        print("✓ ChromaDB funcional")
        return True
    except Exception as e:
        print(f"✗ Erro ChromaDB: {e}")
        return False


def test_embeddings():
    """Testa geração de embeddings"""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode("Teste de embedding")
        print(f"✓ Embeddings funcionais (dimensão: {len(embedding)})")
        return True
    except Exception as e:
        print(f"✗ Erro embeddings: {e}")
        return False


if __name__ == "__main__":
    print("=== Validação ETAPA 2 ===\n")

    success = True
    success &= validate_imports()
    print()
    success &= test_chromadb()
    success &= test_embeddings()

    print("\n" + "=" * 30)
    if success:
        print("✓ Etapa 2 validada com sucesso!")
    else:
        print("✗ Problemas encontrados na validação")
