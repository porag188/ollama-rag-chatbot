"""
PDF Digest Script for Qdrant Vector Database

Usage:
    python digest.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import uuid
from contextlib import suppress

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PDF_FILE_PATH = "./data/cv.pdf"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


class PDFProcessor:
    """Handles PDF text extraction."""

    def __init__(self):
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
        except ImportError:
            logger.error("PyPDF2 not found. Install with: pip install PyPDF2")
            raise

    def extract_text(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                reader = self.PyPDF2.PdfReader(file)
                text_pages = []
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(f"\n--- Page {i + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {i + 1}: {e}")
                text = "".join(text_pages).strip()
                if not text:
                    raise ValueError("No text could be extracted from the PDF")
                logger.info(f"Extracted {len(text)} characters from {len(reader.pages)} pages")
                return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise


class TextChunker:
    """Splits text into overlapping chunks."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)
        chunk_index = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
                "metadata": {"chunk_size": len(chunk_text), "total_chunks": 0}
            })
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        # Set total_chunks in metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = total_chunks

        avg_size = sum(len(c["text"]) for c in chunks) // total_chunks if total_chunks else 0
        logger.info(f"Created {total_chunks} chunks (avg size: {avg_size} chars)")
        return chunks


class PDFDigester:
    """Processes PDFs and stores embeddings in Qdrant."""

    def __init__(self):
        self._cleanup_tasks = []
        self.embedding_service = EmbeddingService(
            ollama_url=settings.ollama_host,
            model_name=settings.ollama_embedding_model
        )
        self.vector_store = VectorStoreService(
            host_url=settings.qdrant_host,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            vector_size=settings.qdrant_vector_size
        )
        self.pdf_processor = PDFProcessor()

    async def digest_pdf(self, pdf_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF")

        logger.info(f"Starting PDF digest for: {pdf_path}")
        await self.vector_store.initialize()

        text = self.pdf_processor.extract_text(pdf_path)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_text(text, source=os.path.basename(pdf_path))
        if not chunks:
            raise ValueError("No chunks created from PDF text")

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        points = []
        for i, chunk in enumerate(chunks, start=1):
            try:
                embedding = await self.embedding_service.embed_text(chunk["text"])
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        **chunk,
                        "pdf_path": pdf_path
                    }
                })
                if i % 10 == 0 or i == len(chunks):
                    logger.info(f"Processed {i}/{len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to embed chunk {i}: {e}")
                continue

        if not points:
            raise RuntimeError("No embeddings generated successfully")

        await self._store_points(points)
        logger.info(f"✅ PDF digested: {pdf_path} ({len(text)} chars, {len(chunks)} chunks, {len(points)} embeddings)")

    async def cleanup(self):
        """Clean up any resources"""
        for task in self._cleanup_tasks:
            with suppress(Exception):
                await task
        self._cleanup_tasks.clear()
    async def _store_points(self, points: List[Dict[str, Any]]):
        from qdrant_client.models import PointStruct
        
        async def get_collection_vector_size(client, collection_name: str) -> int:
            info = await client.get_collection(collection_name=collection_name)
            
            # Extract vector size from the collection config
            if hasattr(info, "config") and hasattr(info.config, "params"):
                params = info.config.params
                if hasattr(params, "vectors"):
                    vectors = params.vectors
                    # Handle different vector configurations
                    if hasattr(vectors, "size"):
                        # Single unnamed vector (your case)
                        return vectors.size
                    elif isinstance(vectors, dict):
                        # Multiple named vectors - take the first one
                        first_vector = list(vectors.values())[0]
                        if hasattr(first_vector, "size"):
                            return first_vector.size
            
            # Fallback: try to access directly from config
            try:
                return info.config.params.vectors.size
            except (AttributeError, KeyError):
                raise ValueError(
                    f"Cannot determine vector size for collection {collection_name}. "
                    "Please check collection config."
                )

        try:
            expected_dim = await get_collection_vector_size(
                self.vector_store.client, self.vector_store.collection_name
            )

            qdrant_points = []
            print('expected_dim', expected_dim)
            for p in points:
                if len(p["vector"]) != expected_dim:
                    logger.warning(
                        f"Skipping point {p['id']}: expected dim {expected_dim}, got {len(p['vector'])}"
                    )
                    continue
                qdrant_points.append(PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]))

            if not qdrant_points:
                logger.warning("No points to store after dimension check.")
                return

            # Batch upload
            batch_size = 100
            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i:i + batch_size]
                await self.vector_store.client.upsert(
                    collection_name=self.vector_store.collection_name,
                    points=batch
                )
                logger.info(
                    f"Stored batch {i // batch_size + 1}/{(len(qdrant_points) + batch_size - 1) // batch_size}"
                )

        except Exception as e:
            logger.error(f"Failed to store points in Qdrant: {e}")
            raise


async def main():
    digester = None
    try:
        digester = PDFDigester()
        await digester.digest_pdf(PDF_FILE_PATH)
    except Exception as e:
        logger.error(e)
        sys.exit(1)
    finally:
        if digester:
            await digester.cleanup()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())

