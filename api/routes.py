from fastapi import APIRouter
from .chat import chat_handler
from models.schemas import (
    ChatRequest,
    ChatResponse,
)

api_router = APIRouter()


@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that handles user queries.
    """
    request_dict = request.model_dump()
    return await chat_handler(request_dict)

