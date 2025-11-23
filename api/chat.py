import logging
from fastapi import HTTPException, status
from config import settings
from services.rag_service import RAGService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.llm_service import LLMService
from services.llm_service import LLMService
from services.query_classifier_service import QueryClassifierService
logger = logging.getLogger(__name__)

async def chat_handler(request: dict):
    """
    Main chat endpoint that handles user queries.
    """
    try:
        user_id = request.get("user_id", "")
        message = request.get("question", "")

        if not message or not message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Question is required"
            )

        logger.info(
            f"Processing chat request: user_id='{user_id}', message='{message[:100]}...'"
        )

        llm_service = LLMService(
            ollama_url=settings.ollama_host, model_name=settings.ollama_llm_model
        )

        query_classifier = QueryClassifierService(llm_service=llm_service)

        use_direct = await query_classifier.should_use_direct_response(message)

        if use_direct:
            logger.info(
                "Query will be answered directly with LLM (greeting or context-based)"
            )

            answer = await query_classifier.generate_direct_response(message)

            return {
                "answer": answer,
                "sources": [],
            }

        else:
            logger.info("Query requires document search, using RAG pipeline")

            embedding_service = EmbeddingService(
                ollama_url=settings.ollama_host,
                model_name=settings.ollama_embedding_model,
            )

            vector_store = VectorStoreService(
                host_url=settings.qdrant_host,
                api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection_name,
                vector_size=settings.qdrant_vector_size,
            )

            rag_service = RAGService(
                embedding_service=embedding_service,
                vector_store=vector_store,
                llm_service=llm_service,
                min_similarity_threshold=settings.similarity_threshold,
                fallback_message=settings.rag_fallback_message,
            )

            rag_response = await rag_service.get_answer(message)

            answer = rag_response["answer"]

            return {
                "answer": answer,
                "sources": rag_response["sources"],
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
