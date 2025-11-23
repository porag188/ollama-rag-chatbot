import logging
from typing import Dict
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


class QueryClassifierService:
    """
    Service for classifying user queries to determine if they need RAG pipeline.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initialize the QueryClassifierService.
        """
        self.llm_service = llm_service

    async def should_use_direct_response(self, query: str) -> bool:
        """
        Decide if a query should be answered directly or requires the RAG pipeline.
        """
        if not query or not query.strip():
            return True  # empty → direct response

        decision_prompt = (
            "You are a decision classifier. Decide whether the user's query should be:\n"
            "- DIRECT → Answered immediately by the assistant (greetings, chit-chat, general knowledge, "
            "questions about the assistant, opinion questions, simple factual questions)\n"
            "- RAG → Requires searching internal documents (company policies, pricing, services, procedures, "
            "product details, account-related questions, technical instructions, configuration details, etc.)\n\n"
            f"User Query: \"{query}\"\n\n"
            "Rules:\n"
            "Use DIRECT when:\n"
            " - The query is general conversation (hello, how are you, who are you, what can you do)\n"
            " - It is general knowledge (what is python, what is AI)\n"
            " - It is opinion/creative (tell me a joke, explain something)\n"
            " - It does NOT mention company-specific or product-specific details\n\n"
            "Use RAG when:\n"
            " - The query asks about company information, policies, pricing, packages, billing\n"
            " - Product or system features, configuration, APIs, errors, troubleshooting\n"
            " - Anything requiring factual accuracy about stored knowledge\n\n"
            "Respond with ONLY one word: DIRECT or RAG.\n\n"
            "Decision:"
        )

        try:
            decision = await self.llm_service.generate_answer(decision_prompt, [])
            decision = decision.strip().upper()

            use_direct = (decision == "DIRECT")

            logger.info(
                f"LLM decision for query '{query[:50]}...': {decision} "
                f"(use_direct={use_direct})"
            )

            return use_direct

        except Exception as e:
            logger.error(f"Error in LLM decision making: {str(e)}")
            return False

    async def generate_direct_response(self, query: str) -> str:
        """
        Generate a direct response using LLM for greetings and chit-chat.

        Args:
            query: User's query

        Returns:
            Generated response text
        """
        direct_prompt = (
            "You are a friendly and helpful assistant. Respond naturally to the user's message.\n\n"
            f"User: {query}\n\n"
            "Assistant:"
        )

        try:
            response = await self.llm_service.generate_answer(direct_prompt, [])

            logger.info(f"Generated direct response (length={len(response)})")

            return response

        except Exception as e:
            logger.error(f"Error generating direct response: {str(e)}")
            return "Hello! How can I help you today?"
