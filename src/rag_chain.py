"""RAG Chain - Retrieval + Generation"""

# Fix SQLite para ChromaDB
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from typing import List, Dict
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()


class RAGChain:
    """Chain RAG para recuperação e geração de respostas"""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        top_k: int = 3,
        llm_provider: str = "ollama",  # "ollama" ou "openai"
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.llm_provider = llm_provider

        # Carrega ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' carregada")
        except:
            raise ValueError(
                f"Collection '{collection_name}' não encontrada. Execute o document_processor primeiro."
            )

        # Modelo de embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Inicializa LLM
        self._setup_llm()

    def _setup_llm(self):
        """Configura o LLM baseado no provider"""
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não encontrada no .env")

            from langchain_community.llms import OpenAI

            self.llm = OpenAI(temperature=0.7, openai_api_key=api_key)
            print("✓ OpenAI configurado")

        elif self.llm_provider == "ollama":
            from langchain_community.llms import Ollama

            self.llm = Ollama(
                model="llama2", temperature=0.7  # ou outro modelo instalado
            )
            print("✓ Ollama configurado (certifique-se que está rodando)")
        else:
            raise ValueError(f"Provider não suportado: {self.llm_provider}")

    def retrieve(self, query: str) -> List[Dict]:
        """Recupera chunks relevantes"""
        # Gera embedding da query
        query_embedding = self.embedding_model.encode([query])[0]

        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=self.top_k
        )

        # Formata resultados
        retrieved_docs = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            retrieved_docs.append({"content": doc, "metadata": metadata, "rank": i + 1})

        return retrieved_docs

    def generate_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Gera prompt para o LLM com contexto"""
        context = "\n\n".join(
            [f"[Documento {doc['rank']}]\n{doc['content']}" for doc in context_docs]
        )

        prompt = f"""Você é um assistente especializado que responde perguntas baseado em documentos fornecidos.

CONTEXTO:
{context}

PERGUNTA: {query}

INSTRUÇÕES:
- Responda a pergunta usando APENAS as informações do contexto acima
- Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos"
- Seja claro e objetivo
- Cite o número do documento quando relevante

RESPOSTA:"""

        return prompt

    def query(self, question: str, return_context: bool = False) -> Dict:
        """Executa query RAG completa"""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print("=" * 60)

        # 1. Retrieval
        print(f"\n[1/2] Recuperando contexto (top {self.top_k})...")
        context_docs = self.retrieve(question)

        print(f"✓ {len(context_docs)} documentos recuperados")
        for doc in context_docs:
            preview = doc["content"][:100].replace("\n", " ")
            print(f"  [{doc['rank']}] {preview}...")

        # 2. Generation
        print(f"\n[2/2] Gerando resposta...")
        prompt = self.generate_prompt(question, context_docs)

        try:
            answer = self.llm.invoke(prompt)
            print("✓ Resposta gerada")
        except Exception as e:
            answer = f"Erro ao gerar resposta: {e}"
            print(f"✗ Erro: {e}")

        result = {
            "question": question,
            "answer": answer,
            "context_used": len(context_docs),
        }

        if return_context:
            result["context_docs"] = context_docs

        return result

    def chat(self):
        """Interface interativa de chat"""
        print("\n" + "=" * 60)
        print("RAG Chat - Digite 'sair' para encerrar")
        print("=" * 60)

        while True:
            try:
                question = input("\n🤔 Pergunta: ").strip()

                if question.lower() in ["sair", "exit", "quit"]:
                    print("👋 Até logo!")
                    break

                if not question:
                    continue

                result = self.query(question)
                print(f"\n💡 Resposta:\n{result['answer']}")

            except KeyboardInterrupt:
                print("\n👋 Até logo!")
                break
            except Exception as e:
                print(f"Erro: {e}")


if __name__ == "__main__":
    print("RAG Chain - Use de forma interativa ou programática")
    print("\nExemplo de uso:")
    print(
        "  rag = RAGChain(collection_name='test_collection', persist_directory='./test_chroma_db')"
    )
    print("  result = rag.query('Sua pergunta aqui')")
    print("  # ou")
    print("  rag.chat()  # modo interativo")
