import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat import chat_handler
from services.flow_matcher import FlowMatcherService
from config import settings


async def test_flow_matching():
    print("=== Flow Matching Tests ===")

    test_cases = [
        {"input": {"user_id": "test_user", "question": "Hello"}, "language": "English"},
        {"input": {"user_id": "test_user", "question": "হ্যালো"}, "language": "Bengali"},
        {"input": {"user_id": "test_user", "question": "Hlw"}, "language": "Banglish"},
    ]

    flow_matcher = FlowMatcherService(settings.file_config_path)

    for case in test_cases:
        trigger_id = flow_matcher.match_flow(case["input"]["question"])
        print(f"flow: {case['input']['question']} ({case['language']})")
        print(f"response: {trigger_id or 'None'}")
        print(f"passed: {trigger_id is not None}\n")


async def test_chat_handler_end_to_end():
    print("=== Chat Handler End-to-End Test ===")

    test_cases = [
        {"user_id": "test_user", "question": "Hello", "language": "English"},
        {"user_id": "test_user", "question": "হ্যালো", "language": "Bengali"},
        {"user_id": "test_user", "question": "Hlw", "language": "Banglish"},
    ]

    for case in test_cases:
        try:
            response = await chat_handler(case)

            answer = response.get("answer", "")

            print(f"flow: {case['question']} ({case['language']})")
            print(f"response: {answer[:100]}")

        except Exception as e:
            print(f"flow: {case['question']} ({case['language']})")
            print(f"response: Error - {str(e)[:100]}")
            print("passed: False\n")


def init_rag_services():
    """Helper function to initialize RAG-related services."""
    from services.embedding_service import EmbeddingService
    from services.vector_store_service import VectorStoreService
    from services.llm_service import LLMService
    from services.rag_service import RAGService

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

    llm_service = LLMService(
        ollama_url=settings.ollama_host,
        model_name=settings.ollama_llm_model,
    )

    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        min_similarity_threshold=settings.similarity_threshold,
        fallback_message=settings.rag_fallback_message,
    )

    return rag_service


async def run_rag_tests(test_queries, check_sources=False):
    rag_service = init_rag_services()
    for item in test_queries:
        query = item["query"]
        language = item["language"]
        try:
            response = await rag_service.get_answer(query)
            answer = response.get("answer", "")
            sources = response.get("sources", [])

            print(f"query: {query} ({language})")
            print(f"response: {answer[:100]}...")
            if check_sources:
                print(f"passed: {len(sources) == 0}\n")
            else:
                print(f"passed: {len(answer) > 0}\n")

        except Exception as e:
            print(f"query: {query} ({language})")
            print(f"response: Error - {str(e)[:100]}")
            print("passed: False\n")


async def main():
    print("Codeware RAG Chatbot Component Tests\n")

    try:
        await test_flow_matching()
        await test_chat_handler_end_to_end()

        # RAG tests
        rag_test_queries = [{"query": "Tell me about your skills", "language": "English"}]
        await run_rag_tests(rag_test_queries)

        # Fallback mechanism tests
        fallback_test_queries = [
            {"query": "where is capital of bangladesh?", "language": "English"},
            {"query": "বাংলাদেশের রাজধানী কোথায়?", "language": "Bengali"},
            {"query": "bangladesh er rajdhani kothay?", "language": "Banglish"},
        ]
        await run_rag_tests(fallback_test_queries, check_sources=True)

        print("All tests completed!")

    except Exception as e:
        print(f"Tests failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
