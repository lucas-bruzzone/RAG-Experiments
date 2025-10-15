#!/usr/bin/env python3
"""Validação ETAPA 3 - Processamento de Documentos"""

# Fix SQLite para ChromaDB
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from pathlib import Path

# Cria estrutura de diretórios
os.makedirs("data", exist_ok=True)
os.makedirs("src", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
os.makedirs("tests", exist_ok=True)

# Cria documento de teste
test_file = Path("data/test_document.txt")
if not test_file.exists():
    test_content = """Introdução ao RAG (Retrieval Augmented Generation)

RAG é uma técnica que combina recuperação de informações com geração de linguagem.
O processo envolve três etapas principais:

1. Indexação: Documentos são divididos em chunks e convertidos em embeddings
2. Recuperação: Chunks relevantes são recuperados baseado na similaridade
3. Geração: Um LLM gera resposta usando os chunks recuperados como contexto

Vantagens do RAG:
- Reduz alucinações do modelo
- Permite uso de conhecimento específico do domínio
- Não requer fine-tuning do modelo base
- Conhecimento pode ser atualizado facilmente

Vector Databases:
ChromaDB, Pinecone, Weaviate e FAISS são exemplos de bancos vetoriais
usados para armazenar e recuperar embeddings eficientemente.

Embeddings:
Representações vetoriais densas de texto que capturam significado semântico.
Modelos como sentence-transformers geram embeddings de alta qualidade."""

    test_file.write_text(test_content, encoding="utf-8")
    print(f"✓ Documento de teste criado: {test_file}")

# Importa e testa o processor
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.document_processor import DocumentProcessor

    print("\n=== Validação ETAPA 3 ===\n")

    # Inicializa processor
    processor = DocumentProcessor(
        collection_name="test_collection", persist_directory="./test_chroma_db"
    )

    # Processa documento de teste
    processor.process_file(str(test_file))

    # Verifica estatísticas
    stats = processor.get_collection_stats()
    print(f"\n{'='*50}")
    print("Estatísticas da Collection:")
    print(f"  Nome: {stats['name']}")
    print(f"  Total de chunks: {stats['total_chunks']}")
    print(f"  Diretório: {stats['persist_directory']}")

    # Teste de busca básica
    print(f"\n{'='*50}")
    print("Teste de busca:")
    query_text = "O que é RAG?"
    query_embedding = processor.embedding_model.encode([query_text])[0]

    results = processor.collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=2
    )

    print(f"Query: '{query_text}'")
    print(f"\nTop 2 resultados:")
    for i, doc in enumerate(results["documents"][0], 1):
        print(f"\n{i}. {doc[:200]}...")

    print(f"\n{'='*50}")
    print("✓ Etapa 3 validada com sucesso!")
    print("\nPróximos passos:")
    print("  - Adicione seus PDFs na pasta data/")
    print("  - Use: processor.process_file('data/seu_arquivo.pdf')")

except Exception as e:
    print(f"✗ Erro: {e}")
    import traceback

    traceback.print_exc()
