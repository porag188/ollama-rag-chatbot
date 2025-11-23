"""
FastAPI application entry point for the multilingual RAG chatbot.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from api.routes import api_router
from config import settings
import uvicorn
from PyPDF2 import PdfReader
from qdrant_client.models import VectorParams, Distance

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def verify_ollama_availability() -> bool:
    """
    Verify that Ollama service is available and the required models are installed.
    
    Steps:
    1. Check that Ollama service responds at the specified host.
    2. Retrieve the list of available models.
    3. Ensure both embedding and LLM models are present.
    4. Log helpful warnings and raise exceptions if any required model is missing.
    """
    ollama_url = settings.ollama_host
    embedding_model = settings.ollama_embedding_model
    llm_model = settings.ollama_llm_model

    try:
        # Send GET request to Ollama /api/tags endpoint
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            models_data = response.json()

        # Extract available model names
        available_models = [model.get("name", "") for model in models_data.get("models", [])]
        logger.info(f"✓ Ollama service is available at {ollama_url}")

        # Check for missing models
        missing_models = []
        if embedding_model not in available_models:
            missing_models.append(embedding_model)
            logger.warning(f"  ⚠ Embedding model '{embedding_model}' not found")

        if llm_model not in available_models:
            missing_models.append(llm_model)
            logger.warning(f"  ⚠ LLM model '{llm_model}' not found")

        # If any required models are missing, raise informative exception
        if missing_models:
            install_cmds = ' && '.join([f"ollama pull {m}" for m in missing_models])
            logger.warning(f"  Missing models: {', '.join(missing_models)}")
            logger.warning(f"  Install missing models with: {install_cmds}")
            raise RuntimeError(f"Required Ollama models not found: {', '.join(missing_models)}. "
                               f"Please run: {install_cmds}")

        logger.info(f"✓ Required Ollama models are available")
        return True

    # Connection errors (service not running or network issue)
    except httpx.RequestError as e:
        error_msg = (
            f"Failed to connect to Ollama service at {ollama_url}. "
            f"Error: {str(e)}\n"
            f"Please ensure Ollama is running (e.g., 'ollama serve')."
        )
        logger.error(f"✗ {error_msg}")
        raise ConnectionError(error_msg) from e

    # HTTP errors returned by Ollama
    except httpx.HTTPStatusError as e:
        error_msg = f"Ollama service returned HTTP {e.response.status_code}: {e.response.text}"
        logger.error(f"✗ {error_msg}")
        raise RuntimeError(error_msg) from e

    # Catch-all for unexpected errors
    except Exception as e:
        error_msg = f"Unexpected error verifying Ollama: {str(e)}"
        logger.error(f"✗ {error_msg}")
        raise RuntimeError(error_msg) from e

async def verify_qdrant_connection() -> bool:
    """
    Verify that Qdrant service is available and accessible.
    Checks if the target collection exists.
    """
    qdrant_url = settings.qdrant_host

    try:
        client = AsyncQdrantClient(
            url=settings.qdrant_host,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            timeout=10,
        )

        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]

        logger.info("✓ Qdrant service is available")

        if settings.qdrant_collection_name in collection_names:
            logger.info(f"✓ Target collection '{settings.qdrant_collection_name}' exists")
        else:
            logger.info(
                f"⚠ Collection '{settings.qdrant_collection_name}' not found — creating it..."
            )

            # 👉 Define your vector size & distance (adjust as needed)
            await client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,     # REQUIRED: your embedding size
                    distance=Distance.COSINE,
                )
            )

            logger.info(
                f"✓ Collection '{settings.qdrant_collection_name}' created successfully"
            )

        return True

    except Exception as e:
        error_msg = (
            f"Failed to connect to Qdrant service at {qdrant_url}. "
            f"Error: {str(e)}\n"
        )
        if qdrant_url.startswith(("http://localhost", "http://127.0.0.1")):
            error_msg += "Please ensure Qdrant is running. Start it with: docker run -p 6333:6333 qdrant/qdrant"
        else:
            error_msg += "Please verify your Qdrant URL and API key are correct."

        logger.error(f"✗ {error_msg}")
        raise Exception(error_msg) from e


def verify_flow_config() -> bool:
    """
    Verify that flow configuration file exists and is readable.
    Supports JSON for bot flows and PDF for documents.
    """
    flow_path = Path(settings.file_config_path)
    logger.info(f"Verifying flow configuration at {flow_path}...")

    if not flow_path.exists():
        error_msg = f"Flow configuration file not found: {flow_path}"
        logger.error(f"✗ {error_msg}")
        raise FileNotFoundError(error_msg)

    if not flow_path.is_file():
        error_msg = f"Flow configuration path is not a file: {flow_path}"
        logger.error(f"✗ {error_msg}")
        raise Exception(error_msg)

    try:
        # Handle JSON flow config
        if flow_path.suffix.lower() in [".json"]:
            with open(flow_path, "r", encoding="utf-8") as f:
                flow_data = json.load(f)
            logger.info(f"✓ JSON flow configuration is valid")
            logger.info(f"  Loaded {len(flow_data)} flow entries")

        # Handle PDF file
        elif flow_path.suffix.lower() in [".pdf"]:
            reader = PdfReader(str(flow_path))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info(f"✓ PDF document is readable, extracted {len(text)} characters")

        else:
            error_msg = f"Unsupported file type: {flow_path.suffix}"
            logger.error(f"✗ {error_msg}")
            raise Exception(error_msg)

        return True

    except json.JSONDecodeError as e:
        error_msg = f"Flow configuration file is not valid JSON: {str(e)}"
        logger.error(f"✗ {error_msg}")
        raise Exception(error_msg) from e

    except Exception as e:
        error_msg = f"Failed to read flow configuration: {str(e)}"
        logger.error(f"✗ {error_msg}")
        raise Exception(error_msg) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    logger.info("=" * 60)
    logger.info("Starting RAG Chatbot API...")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Embedding Model: {settings.ollama_embedding_model}")
    logger.info(f"  LLM Model: {settings.ollama_llm_model}")
    logger.info(f"  Collection: {settings.qdrant_collection_name}")
    logger.info(f"  Flow Config: {settings.file_config_path}")
    logger.info("=" * 60)

    try:
        logger.info("Running startup verification checks...")
        logger.info("")

        await verify_ollama_availability()
        logger.info("")

        await verify_qdrant_connection()
        logger.info("")

        verify_flow_config()
        logger.info("")

        logger.info("=" * 60)
        logger.info("✓ All startup checks passed successfully!")
        logger.info("✓ RAG Chatbot API is ready to accept requests")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("✗ STARTUP FAILED - Service dependencies not available")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 60)
        logger.error("The application cannot start without required dependencies.")
        logger.error("Please fix the issues above and restart the application.")
        logger.error("=" * 60)

        sys.exit(1)

    yield

    logger.info("Shutting down RAG Chatbot API...")
    logger.info("Cleanup completed successfully")


app = FastAPI(
    title="Multilingual RAG Chatbot API",
    description="A chatbot API that combines semantic search with rule-based flow triggering",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Codeware - RAG Chatbot API",
        "status": "running",
        "version": "1.0.1",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
