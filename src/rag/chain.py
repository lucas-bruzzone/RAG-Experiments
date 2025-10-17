"""RAG Chain - Combines retrieval and generation"""

from typing import Dict, Optional
from .retriever import Retriever
from ..core.llm_factory import LLMFactory
from ..utils import get_logger, set_correlation_id, timed

logger = get_logger(__name__)


class RAGChain:
    """Complete RAG pipeline: retrieval + generation"""
    
    def __init__(self, config: dict, llm_provider: Optional[str] = None):
        self.config = config
        self.retriever = Retriever(config)
        
        # Initialize LLM
        provider = llm_provider or config.get("llm", {}).get("provider", "auto")
        logger.info(f"Initializing RAGChain with provider: {provider}")
        
        self.llm = LLMFactory.create(config, provider=provider)
        
        logger.info("RAGChain initialized successfully")
    
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
        
        logger.debug(f"Generated prompt with {len(context_docs)} context documents")
        return prompt
    
    @timed
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
        # Set correlation ID for this query
        correlation_id = set_correlation_id()
        
        logger.info(
            f"Processing query: '{question[:100]}...'",
            extra={
                'question_length': len(question),
                'top_k': top_k,
                'correlation_id': correlation_id
            }
        )
        
        try:
            # 1. Retrieval
            logger.debug("Starting retrieval phase")
            context_docs = self.retriever.retrieve(question, top_k=top_k)
            logger.info(f"Retrieved {len(context_docs)} documents")
            
            # 2. Generation
            logger.debug("Starting generation phase")
            prompt = self.generate_prompt(question, context_docs)
            
            try:
                answer = self.llm.invoke(prompt)
                logger.info(
                    "Query completed successfully",
                    extra={
                        'answer_length': len(answer),
                        'context_used': len(context_docs)
                    }
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)
                answer = f"Error generating response: {e}"
            
            result = {
                "question": question,
                "answer": answer,
                "context_used": len(context_docs),
                "correlation_id": correlation_id
            }
            
            if return_context:
                result["context_docs"] = context_docs
            
            return result
            
        except Exception as e:
            logger.error(
                f"Query failed: {e}",
                extra={'question': question},
                exc_info=True
            )
            raise
    
    def chat(self):
        """Interactive chat mode"""
        logger.info("Starting interactive chat mode")
        print("\nRAG Chat - Type 'exit' to quit")
        print("=" * 60)
        
        query_count = 0
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ["exit", "quit", "sair"]:
                    logger.info(f"Chat session ended. Total queries: {query_count}")
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                query_count += 1
                logger.debug(f"Processing query #{query_count}")
                
                print("\nRetrieving context...")
                result = self.query(question, return_context=True)
                
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\n(Used {result['context_used']} documents)")
                
                logger.debug(
                    f"Query #{query_count} completed",
                    extra={'correlation_id': result['correlation_id']}
                )
                
            except KeyboardInterrupt:
                logger.info(f"Chat interrupted by user. Total queries: {query_count}")
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                print(f"Error: {e}")