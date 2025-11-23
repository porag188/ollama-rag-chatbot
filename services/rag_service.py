import logging
from typing import Dict, List, Optional, Any
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        llm_service: LLMService,
        min_similarity_threshold: float = 0.3,
        fallback_message: Optional[str] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.min_similarity_threshold = min_similarity_threshold

        self.fallback_message = fallback_message or (
            "I’m sorry, I don’t have enough information to answer that. "
            "Please try rephrasing or contact support."
        )

    async def get_answer(self, question: str) -> Dict[str, Any]:
        logger.info(f"RAG → Processing: {question[:80]}...")

        try:
            # 1. Embed user question
            query_vector = await self.embedding_service.embed_text(question)

            # 2. Retrieve documents
            results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=5
            )

            # 3. Filter by similarity
            context_docs = [
                r["text"]
                for r in results
                if r.get("score", 0) >= self.min_similarity_threshold
            ]

            sources = list({
                r.get("source", "")
                for r in results
                if r.get("score", 0) >= self.min_similarity_threshold and r.get("source")
            })

            # 4. If no documents → generate personalized sorry message
            if not context_docs:
                sorry_prompt = f"""
                You are a helpful and professional assistant. No relevant documents were found for the user's query.

                User Question: {question}

                Generate a polite, concise 2–3 sentence response that:
                - acknowledges the user’s question,
                - clearly explains that no matching information was found,
                - offers guidance on rephrasing or contacting support for further help.

                Your response should be empathetic, clear, and user-friendly.

                Response:
                """.strip()

                apology = await self.llm_service.generate_answer(sorry_prompt, [])
                return {
                    "answer": apology,
                    "sources": [],
                }

            # 5. Generate final answer using retrieved documents
            answer = await self.llm_service.generate_answer(question, context_docs)

            return {
                "answer": answer,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG pipeline error: {e}") from e
