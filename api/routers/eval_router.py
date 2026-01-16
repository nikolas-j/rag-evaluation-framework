"""Evaluation endpoints for running and viewing results."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel

from eval.runner import run_eval

router = APIRouter()
logger = logging.getLogger(__name__)

RUNS_DIR = Path("storage/runs")

_eval_tasks: Dict[str, Dict[str, Any]] = {}


class EvalRunRequest(BaseModel):
    """Request model for running evaluation."""
    dataset: str = "QA_testing_sets/golden.json"
    run_name: Optional[str] = None
    num_questions: Optional[int] = None
    selected_metrics: Optional[List[str]] = None
    config_overrides: Dict[str, Any] = {}


class EvalRunInfo(BaseModel):
    """Information about an evaluation run."""
    run_id: str
    run_name: str
    timestamp: str
    dataset: str
    status: str  # "running", "completed", "error"
    progress: int = 0  # 0-100
    num_questions: int = 0
    avg_score: Optional[float] = None


class RunSummary(BaseModel):
    """Summary statistics for a run."""
    run_id: str
    run_name: str
    timestamp: str
    total_questions: int
    average_overall_score: float
    metrics: Dict[str, Any]


@router.post("/run")
async def start_evaluation(
    request: Request,
    background_tasks: BackgroundTasks,
    eval_request: EvalRunRequest
):
    """Start an evaluation run in the background.
    
    Returns immediately with run_id. Use WebSocket or /runs/{run_id}
    to monitor progress.
    """
    config_manager = request.app.state.config_manager
    
    # Get settings with overrides
    try:
        if eval_request.config_overrides:
            settings = config_manager.get_config_with_overrides(eval_request.config_overrides)
        else:
            settings = config_manager.get_default_config()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    
    # Generate run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = eval_request.run_name or "eval"
    run_id = f"{timestamp}_{run_name}"
    
    # Initialize task tracking
    _eval_tasks[run_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting evaluation...",
        "dataset": eval_request.dataset,
        "run_name": run_name,
        "timestamp": timestamp,
    }
    
    def run_evaluation_task():
        """Background task to run evaluation."""
        try:
            _eval_tasks[run_id]["message"] = "Running evaluation..."
            _eval_tasks[run_id]["progress"] = 10
            
            # Progress callback to update in-memory task info
            def update_progress(current, total, question):
                _eval_tasks[run_id]["current_question"] = current
                _eval_tasks[run_id]["total_questions"] = total
                _eval_tasks[run_id]["current_question_text"] = question[:100]  # Truncate long questions
            
            # Run evaluation (this is synchronous)
            run_dir = run_eval(
                dataset_path=eval_request.dataset,
                settings=settings,
                run_name=run_name,
                num_questions=eval_request.num_questions,
                selected_metrics=eval_request.selected_metrics,
                progress_callback=update_progress,
            )
            
            _eval_tasks[run_id]["status"] = "completed"
            _eval_tasks[run_id]["progress"] = 100
            _eval_tasks[run_id]["message"] = "Evaluation completed"
            _eval_tasks[run_id]["run_dir"] = str(run_dir)
            
            logger.info(f"Evaluation {run_id} completed: {run_dir}")
        except Exception as e:
            _eval_tasks[run_id]["status"] = "error"
            _eval_tasks[run_id]["message"] = f"Error: {str(e)}"
            _eval_tasks[run_id]["progress"] = 0
            logger.error(f"Evaluation {run_id} failed: {e}")
    
    background_tasks.add_task(run_evaluation_task)
    
    return {
        "run_id": run_id,
        "status": "started",
        "message": "Evaluation started in background"
    }


@router.get("/runs", response_model=List[EvalRunInfo])
async def list_runs():
    """List all evaluation runs.
    
    Returns metadata for all runs in storage/runs/.
    """
    if not RUNS_DIR.exists():
        return []
    
    runs = []
    
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        run_id = run_dir.name
        
        # Check if this is a running task
        if run_id in _eval_tasks:
            task_info = _eval_tasks[run_id]
            runs.append(EvalRunInfo(
                run_id=run_id,
                run_name=task_info.get("run_name", ""),
                timestamp=task_info.get("timestamp", ""),
                dataset=task_info.get("dataset", ""),
                status=task_info["status"],
                progress=task_info.get("progress", 0)
            ))
            continue
        
        # Load completed run info
        try:
            metadata_file = run_dir / "metadata.json"
            config_snapshot = run_dir / "config_snapshot.json"
            report_jsonl = run_dir / "report.jsonl"
            
            # Try to load metadata first (new format)
            dataset_name = ""
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    dataset_name = metadata.get("dataset_name", "")
                    timestamp_str = metadata.get("timestamp_utc", "")
                    run_name = metadata.get("run_name", run_id)
            else:
                # Fall back to parsing run_id
                parts = run_id.split("_", 2)
                if len(parts) >= 3:
                    timestamp_str = f"{parts[0]}_{parts[1]}"
                    run_name = parts[2]
                else:
                    timestamp_str = ""
                    run_name = run_id
            
            # Calculate summary stats if report exists
            avg_score = None
            num_questions = 0
            
            if report_jsonl.exists():
                with open(report_jsonl, 'r') as f:
                    records = [json.loads(line) for line in f]
                    num_questions = len(records)
                    
                    if records:
                        scores = [r.get("overall_score", 0) for r in records]
                        avg_score = sum(scores) / len(scores) if scores else 0
            
            runs.append(EvalRunInfo(
                run_id=run_id,
                run_name=run_name,
                timestamp=timestamp_str,
                dataset=dataset_name,
                status="completed",
                progress=100,
                num_questions=num_questions,
                avg_score=avg_score
            ))
        except Exception as e:
            logger.warning(f"Failed to load run {run_id}: {e}")
            continue
    
    return runs


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    """Get detailed information about a specific run.
    
    Returns config, summary, and full results.
    """
    run_dir = RUNS_DIR / run_id
    
    # Check if run directory exists
    if not run_dir.exists():
        # If not on disk yet, check if it's a newly started run
        if run_id in _eval_tasks:
            task_info = _eval_tasks[run_id]
            return {
                "run_id": run_id,
                "status": task_info["status"],
                "message": task_info.get("message", "Starting evaluation..."),
                "current_question": task_info.get("current_question", 0),
                "total_questions": task_info.get("total_questions", 0),
                "current_question_text": task_info.get("current_question_text", ""),
                "metadata": {},
                "results": []
            }
        raise HTTPException(status_code=404, detail="Run not found")
    
    try:
        result = {"run_id": run_id}
        
        # Load metadata (available immediately when run starts)
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                result["metadata"] = metadata
                result["run_name"] = metadata.get("run_name", "")
                result["dataset"] = metadata.get("dataset_path", "")
        else:
            result["metadata"] = {}
        
        # Load config snapshot
        config_path = run_dir / "config_snapshot.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                result["config"] = json.load(f)
        
        # Load summary (only available after completion)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                result["summary"] = json.load(f)
        
        # Load summary markdown
        summary_md_path = run_dir / "summary.md"
        if summary_md_path.exists():
            result["summary_md"] = summary_md_path.read_text(encoding='utf-8')
        
        # Load report JSONL (grows during evaluation)
        report_path = run_dir / "report.jsonl"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                result["results"] = [json.loads(line) for line in f if line.strip()]
        else:
            result["results"] = []
        
        # Check if still running
        if run_id in _eval_tasks:
            task_info = _eval_tasks[run_id]
            result["status"] = task_info["status"]
        else:
            result["status"] = "completed"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load run: {str(e)}")


@router.get("/runs/{run_id}/download")
async def download_run(run_id: str):
    """Download a run's summary.md file.
    
    Future: Could be extended to download ZIP of all files.
    """
    run_dir = RUNS_DIR / run_id
    summary_path = run_dir / "summary.md"
    
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return FileResponse(
        path=summary_path,
        filename=f"{run_id}_summary.md",
        media_type="text/markdown"
    )


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete an evaluation run."""
    run_dir = RUNS_DIR / run_id
    
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Don't allow deleting running evaluations
    if run_id in _eval_tasks and _eval_tasks[run_id]["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete running evaluation")
    
    try:
        import shutil
        shutil.rmtree(run_dir)
        
        # Remove from tasks if present
        if run_id in _eval_tasks:
            del _eval_tasks[run_id]
        
        return {"message": f"Deleted run: {run_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-metrics")
async def get_metrics_list():
    """Get list of all available evaluation metrics from registry."""
    from eval.metrics import METRIC_REGISTRY, DEFAULT_METRICS
    
    return {
        "metrics": list(METRIC_REGISTRY.keys()),
        "default": DEFAULT_METRICS
    }


@router.websocket("/ws/{run_id}")
async def eval_websocket(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time evaluation progress."""
    await websocket.accept()
    
    try:
        while True:
            if run_id not in _eval_tasks:
                await websocket.send_json({
                    "status": "error",
                    "message": "Run not found"
                })
                break
            
            task_info = _eval_tasks[run_id]
            await websocket.send_json(task_info)
            
            if task_info["status"] in ["completed", "error"]:
                break
            
            await asyncio.sleep(0.5)
        
        await websocket.close()
    except WebSocketDisconnect:
        logger.info(f"Evaluation WebSocket disconnected for {run_id}")
