"""System endpoints for health and maintenance."""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SystemStatus(BaseModel):
    """System status information."""
    vector_store_exists: bool
    knowledge_base_files: int
    datasets: int
    eval_runs: int


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and statistics."""
    vector_store_dir = Path("storage/chroma")
    kb_dir = Path("knowledge_base")
    datasets_dir = Path("QA_testing_sets")
    runs_dir = Path("storage/runs")
    
    # Count files
    kb_files = len(list(kb_dir.rglob("*.txt"))) if kb_dir.exists() else 0
    dataset_files = len(list(datasets_dir.glob("*.json"))) if datasets_dir.exists() else 0
    run_dirs = len([d for d in runs_dir.iterdir() if d.is_dir()]) if runs_dir.exists() else 0
    
    return SystemStatus(
        vector_store_exists=vector_store_dir.exists(),
        knowledge_base_files=kb_files,
        datasets=dataset_files,
        eval_runs=run_dirs
    )


@router.post("/reset")
async def reset_vector_store():
    """Reset the vector store (deletes all indexed documents)."""
    vector_store_dir = Path("storage/chroma")
    
    if not vector_store_dir.exists():
        return {"message": "Vector store does not exist"}
    
    try:
        shutil.rmtree(vector_store_dir)
        return {"message": "Vector store reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")
