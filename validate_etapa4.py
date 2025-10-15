#!/usr/bin/env python3
"""Validação ETAPA 4 - RAG Chain (Retrieval + Generation)"""

# Fix SQLite para ChromaDB
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

print("=== Validação ETAPA 4 ===\n")

# Teste 1: Retrieval (sem LLM)
print("[Teste 1] Retrieval isolado...")
try:
    from src.document_processor import DocumentProcessor
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Carrega collection de teste
    client = chromadb.PersistentClient(path="./test_chroma_db")
    collection = client.get_collection("test_collection")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Teste de retrieval
    query = "Quais são as vantagens do RAG?"
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=2)

    print(f"✓ Retrieval funcional")
    print(f"  Query: '{query}'")
    print(f"  Documentos recuperados: {len(results['documents'][0])}")

except Exception as e:
    print(f"✗ Erro no retrieval: {e}")
    sys.exit(1)

# Teste 2: RAG Chain (requer LLM)
print(f"\n{'='*60}")
print("[Teste 2] RAG Chain completa (requer LLM)...")

try:
    from src.rag_chain import RAGChain

    # Verifica se Ollama está disponível
    import subprocess

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        ollama_available = result.returncode == 0
    except:
        ollama_available = False

    if ollama_available:
        print("\n✓ Ollama detectado")
        print("\nTestando RAG completo...")

        rag = RAGChain(
            collection_name="test_collection",
            persist_directory="./test_chroma_db",
            llm_provider="ollama",
        )

        result = rag.query("O que é RAG?")

        print(f"\n{'='*60}")
        print("RESULTADO:")
        print(f"  Pergunta: {result['question']}")
        print(f"  Contexto usado: {result['context_used']} documentos")
        print(f"  Resposta: {result['answer'][:200]}...")
        print(f"{'='*60}")
        print("\n✓ Etapa 4 validada com sucesso!")

    else:
        print("\n⚠ Ollama não detectado")
        print("\nOpções para usar o LLM:")
        print("\n1. Ollama (recomendado para testes):")
        print("   - Instale: https://ollama.ai")
        print("   - Execute: ollama run llama2")
        print("   - Teste novamente")

        print("\n2. OpenAI API:")
        print("   - Crie .env com: OPENAI_API_KEY=sua_chave")
        print("   - Use: RAGChain(llm_provider='openai')")

        print("\n✓ Retrieval validado (Generation requer LLM)")

except ImportError as e:
    print(f"✗ Erro de importação: {e}")
    print("\nVerifique se todos os arquivos foram criados:")
    print("  - src/document_processor.py")
    print("  - src/rag_chain.py")
except Exception as e:
    print(f"✗ Erro: {e}")
    import traceback

    traceback.print_exc()

print(f"\n{'='*60}")
print("Próximos passos:")
print("  1. Configure um LLM (Ollama ou OpenAI)")
print("  2. Use a RAG chain:")
print("     from src.rag_chain import RAGChain")
print("     rag = RAGChain()")
print("     rag.chat()  # modo interativo")
