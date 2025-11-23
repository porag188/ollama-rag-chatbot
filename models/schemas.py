from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.

    Attributes:
        user_id: Unique identifier for the user
        question: User's question in English, Bangla, or Banglish
    """

    user_id: str = Field(..., description="Unique identifier for the user")
    question: str = Field(..., description="User's question text")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Validate that user_id is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("user_id cannot be empty")
        return v.strip()

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate that question is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_id like 123",
                    "question": "What is the loan application process?",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.

    Attributes:
        answer: Generated answer or flow response
        sources: List of source document references (empty for flow responses)
    """

    answer: str = Field(..., description="Generated answer or flow response")
    sources: List[str] = Field(
        default_factory=list, description="List of source document references"
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Based on the documents, the loan application process involves...",
                    "sources": ["document1.pdf", "document2.pdf"],
                },
                {
                    "answer": "Flow triggered for loan application service",
                    "sources": [],
                },
            ]
        }
    }


