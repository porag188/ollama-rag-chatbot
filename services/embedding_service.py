import logging
from typing import List
import httpx

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Ollama embedding models.
    """

    def __init__(self, ollama_url: str, model_name: str, timeout: int = 60):
        """
        Initialize the EmbeddingService.
        """
        self.base_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout = httpx.Timeout(timeout)
        self._embedding_dimension: int | None = None

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for embedding generation")

        endpoint = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}

        logger.debug(f"Generating embedding for text (length={len(text)})")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                embedding = response.json().get("embedding")

                if not embedding:
                    raise ValueError("Ollama returned empty embedding")

                if self._embedding_dimension is None:
                    self._embedding_dimension = len(embedding)

                logger.debug(f"Embedding generated successfully (dimension={len(embedding)})")
                return embedding

        except httpx.TimeoutException as e:
            msg = f"Ollama embedding timeout: Request exceeded {self.timeout.read}s"
            logger.error(msg)
            raise RuntimeError(msg) from e

        except httpx.HTTPStatusError as e:
            msg = f"Ollama API error {e.response.status_code}: {e.response.text}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        except httpx.RequestError as e:
            msg = f"Ollama connection error: {e}. Is Ollama running at {self.base_url}?"
            logger.error(msg)
            raise RuntimeError(msg) from e

        except Exception as e:
            logger.error(f"Unexpected error while generating embedding: {e}")
            raise RuntimeError(f"Unexpected error generating embedding: {e}") from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension size of the embedding vectors.
        """
        if self._embedding_dimension is None:
            raise RuntimeError(
                "Embedding dimension unknown. Generate at least one embedding first."
            )
        return self._embedding_dimension
