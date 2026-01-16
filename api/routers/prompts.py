"""Prompt management endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_app.prompt_manager import get_prompt_manager

router = APIRouter()


class PromptCreateRequest(BaseModel):
    """Request model for creating/updating prompts."""
    title: str
    content: str
    description: str = ""
    metric: Optional[str] = None  # For eval prompts
    type: str = "system"  # For RAG prompts


class PromptResponse(BaseModel):
    """Response model for prompt data."""
    title: str
    content: str
    description: str = ""
    created_at: str
    modified_at: Optional[str] = None
    metric: Optional[str] = None  # For eval prompts
    type: Optional[str] = None  # For RAG prompts


@router.get("/{category}", response_model=List[PromptResponse])
async def list_prompts(category: str):
    """List all prompts for a category.
    
    Args:
        category: "rag" or "eval"
    """
    if category not in ["rag", "eval"]:
        raise HTTPException(status_code=400, detail="Category must be 'rag' or 'eval'")
    
    prompt_manager = get_prompt_manager()
    prompts = prompt_manager.list_prompts(category)
    
    return [PromptResponse(**p) for p in prompts]


@router.get("/{category}/{title}")
async def get_prompt(category: str, title: str, metric: Optional[str] = None):
    """Get a specific prompt by title.
    
    Args:
        category: "rag" or "eval"
        title: Prompt title
        metric: For eval prompts, the metric name
    """
    if category not in ["rag", "eval"]:
        raise HTTPException(status_code=400, detail="Category must be 'rag' or 'eval'")
    
    prompt_manager = get_prompt_manager()
    content = prompt_manager.get_prompt(category, title, metric)
    
    if not content:
        raise HTTPException(status_code=404, detail=f"Prompt '{title}' not found")
    
    # Find full prompt data
    prompts = prompt_manager.list_prompts(category)
    for p in prompts:
        if category == "eval":
            if p.get("title") == title and p.get("metric") == metric:
                return PromptResponse(**p)
        else:
            if p.get("title") == title:
                return PromptResponse(**p)
    
    raise HTTPException(status_code=404, detail=f"Prompt '{title}' not found")


@router.post("/{category}")
async def save_prompt(category: str, prompt_data: PromptCreateRequest):
    """Save a new prompt or update existing.
    
    Args:
        category: "rag" or "eval"
        prompt_data: Prompt details
    """
    if category not in ["rag", "eval"]:
        raise HTTPException(status_code=400, detail="Category must be 'rag' or 'eval'")
    
    if category == "eval" and not prompt_data.metric:
        raise HTTPException(status_code=400, detail="Metric name required for eval prompts")
    
    prompt_manager = get_prompt_manager()
    success = prompt_manager.save_prompt(
        category=category,
        title=prompt_data.title,
        content=prompt_data.content,
        description=prompt_data.description,
        metric=prompt_data.metric,
        prompt_type=prompt_data.type
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save prompt")
    
    return {"success": True, "message": f"Prompt '{prompt_data.title}' saved successfully"}


@router.delete("/{category}/{title}")
async def delete_prompt(category: str, title: str, metric: Optional[str] = None):
    """Delete a prompt.
    
    Args:
        category: "rag" or "eval"
        title: Prompt title
        metric: For eval prompts, the metric name
    """
    if category not in ["rag", "eval"]:
        raise HTTPException(status_code=400, detail="Category must be 'rag' or 'eval'")
    
    # Prevent deletion of default prompts
    if title == "Default v1.0":
        raise HTTPException(status_code=400, detail="Cannot delete default prompts")
    
    prompt_manager = get_prompt_manager()
    success = prompt_manager.delete_prompt(category, title, metric)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Prompt '{title}' not found")
    
    return {"success": True, "message": f"Prompt '{title}' deleted successfully"}
