import logging
from typing import List
import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for generating answers using an Ollama model.
    """

    def __init__(self, ollama_url: str, model_name: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout = httpx.Timeout(120)
        self.endpoint = f"{self.ollama_url}/api/generate"

    async def generate_answer(self, question: str, context_docs: List[str]) -> str:
        """
        Generate an answer using Ollama with or without context.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Build prompt
        if context_docs:
            context_text = "\n\n".join(
                f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)
            )
            prompt = (
                "You are a helpful assistant. Use ONLY the provided documents.\n\n"
                f"{context_text}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        else:
            prompt = (
                "You are a helpful assistant. Answer the question directly.\n\n"
                f"Question: {question}\n\nAnswer:"
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        logger.debug(
            f"LLM Request → model={self.model_name}, "
            f"context_docs={len(context_docs)}, "
            f"question_length={len(question)}"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()

                data = response.json()
                answer = data.get("response", "").strip()

                if not answer:
                    raise RuntimeError("Ollama returned an empty response.")

                return answer

        except httpx.TimeoutException as e:
            logger.error("Ollama request timed out", exc_info=e)
            raise TimeoutError("Ollama took too long to respond.") from e

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ollama API HTTP error: {e.response.status_code}", exc_info=e
            )
            raise RuntimeError(
                f"Ollama HTTP error: {e.response.status_code} - {e.response.text}"
            ) from e

        except httpx.RequestError as e:
            logger.error("Ollama connection error", exc_info=e)
            raise ConnectionError(
                f"Could not reach Ollama at {self.ollama_url}. Is it running?"
            ) from e

        except Exception as e:
            logger.error("Unexpected error in LLMService", exc_info=e)
            raise RuntimeError(f"Unexpected LLM error: {e}") from e
