"""
Configuration management using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_host: str = Field("http://localhost:11434", description="Ollama service URL with port")
    ollama_embedding_model: str = Field("embeddinggemma:latest", description="Ollama embedding model name")
    ollama_llm_model: str = Field("granite3.1-moe:1b", description="Ollama LLM model name")

    # Qdrant Configuration
    qdrant_host: str = Field("http://localhost:6333", description="Qdrant service URL with port")
    qdrant_api_key: str = Field("", description="Qdrant API key (optional)")
    qdrant_collection_name: str = Field("documents", description="Qdrant collection name")
    qdrant_vector_size: int = Field(768, description="Vector embedding dimension size")

    # file Configuration
    file_config_path: str = Field("data/cv.pdf", description="Path to data configuration file")

    # RAG Configuration
    similarity_threshold: float = Field(0.3, description="Minimum similarity score to consider a document relevant for RAG")
    rag_fallback_message: str = Field(
        (
            "Sorry, I couldn’t find enough relevant information in my knowledge base to answer that right now. "
            "You can try asking the question in a different way or be a bit more specific about what you need. "
            "If your question is related to our services—such as packages, billing, account issues, or technical support—"
            "I’ll be happy to help with those as well."
        ),
        description="Fallback message when no relevant documents are found",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
