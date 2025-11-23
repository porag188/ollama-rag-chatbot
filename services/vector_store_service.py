import logging
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams


logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing Qdrant vector database operations."""

    def __init__(self, host_url: str, api_key: str, collection_name: str, vector_size: int):
        """
        Initialize VectorStoreService with Qdrant connection parameters.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = AsyncQdrantClient(url=host_url, api_key=api_key or None, timeout=30)

    async def initialize(self) -> None:
        """
        Initialize the vector store by verifying if the collection exists, otherwise create it.
        """
        try:
            collections = await self.client.get_collections()
            if self.collection_name in (col.name for col in collections.collections):
                logger.info(f"Collection '{self.collection_name}' already exists")
            else:
                logger.info(f"Collection '{self.collection_name}' not found. Creating new collection...")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(
                    f"Collection '{self.collection_name}' created with vector size {self.vector_size} "
                    f"and COSINE distance metric"
                )
        except Exception as e:
            logger.exception(f"Failed to initialize collection '{self.collection_name}'")
            raise

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search against the Qdrant collection.
        """
        try:
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            results = [
                {"text": hit.payload.get("text", ""), "source": hit.payload.get("source", ""), "score": hit.score}
                for hit in search_results
            ]
            logger.info(f"Retrieved {len(results)} documents from collection '{self.collection_name}'")
            return results
        except Exception as e:
            logger.exception(f"Failed to search collection '{self.collection_name}'")
            raise