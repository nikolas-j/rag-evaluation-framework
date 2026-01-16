"""Query endpoint for answering questions."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rag_app.rag import answer_question

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for answering a question."""
    question: str
    top_k: Optional[int] = None
    # Allow any additional config overrides
    config_overrides: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Response model for query results."""
    question: str
    answer: str
    contexts: list
    sources: list
    prompt_version: str
    config_snapshot: Dict[str, Any]
    retrieval_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@router.post("", response_model=QueryResponse)
async def answer_query(request: Request, query: QueryRequest):
    """Answer a question using the RAG pipeline.
    
    Applies configuration overrides (like top_k) per-request,
    mimicking CLI parameter behavior.
    """
    config_manager = request.app.state.config_manager
    
    # Build config overrides
    overrides = query.config_overrides.copy()
    if query.top_k is not None:
        overrides["top_k"] = query.top_k
    
    # Get settings with overrides applied
    try:
        if overrides:
            settings = config_manager.get_config_with_overrides(overrides)
        else:
            settings = config_manager.get_default_config()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    
    # Answer question
    try:
        result = answer_question(
            question=query.question,
            settings=settings,
            top_k=query.top_k
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
