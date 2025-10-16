"""RAG Chain - Combines retrieval and generation"""

from typing import Dict, Optional
from .retriever import Retriever
from ..core.llm_factory import LLMFactory


class RAGChain:
    """Complete RAG pipeline: retrieval + generation"""
    
    def __init__(self, config: dict, llm_provider: Optional[str] = None):
        self.config = config
        self.retriever = Retriever(config)
        
        # Initialize LLM
        provider = llm_provider or config.get("llm", {}).get("provider", "auto")
        self.llm = LLMFactory.create(config, provider=provider)
    
    def generate_prompt(self, query: str, context_docs: list) -> str:
        """Generate prompt with context"""
        context = "\n\n".join([
            f"[Document {doc['rank']}]\n{doc['content']}"
            for doc in context_docs
        ])
        
        prompt = f"""You are a specialized assistant that answers questions based on provided documents.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer the question using ONLY information from the context above
- If the answer is not in the context, say "I don't have that information in the documents"
- Be clear and objective
- Cite document numbers when relevant

ANSWER:"""
        
        return prompt
    
    def query(self, question: str, return_context: bool = False, top_k: Optional[int] = None) -> Dict:
        """
        Execute complete RAG query
        
        Args:
            question: User question
            return_context: Include context docs in response
            top_k: Override default top_k
        
        Returns:
            Dict with: question, answer, context_used, [context_docs]
        """
        # 1. Retrieval
        context_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # 2. Generation
        prompt = self.generate_prompt(question, context_docs)
        
        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            answer = f"Error generating response: {e}"
        
        result = {
            "question": question,
            "answer": answer,
            "context_used": len(context_docs)
        }
        
        if return_context:
            result["context_docs"] = context_docs
        
        return result
    
    def chat(self):
        """Interactive chat mode"""
        print("\nRAG Chat - Type 'exit' to quit")
        print("=" * 60)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ["exit", "quit", "sair"]:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nRetrieving context...")
                result = self.query(question, return_context=True)
                
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\n(Used {result['context_used']} documents)")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
