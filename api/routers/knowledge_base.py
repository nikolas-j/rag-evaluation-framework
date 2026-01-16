"""Knowledge base management endpoints."""

import logging
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from rag_app.ingest import ingest_knowledge_base

router = APIRouter()
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path("knowledge_base")


class FileInfo(BaseModel):
    """Information about a knowledge base file."""
    path: str
    category: str
    filename: str
    size: int


class FileTreeNode(BaseModel):
    """Tree node for file browser."""
    name: str
    type: str  # "file" or "directory"
    path: str
    children: List["FileTreeNode"] = []
    size: int = 0


class IngestStatus(BaseModel):
    """Status of ingestion operation."""
    status: str  # "idle", "running", "completed", "error"
    message: str = ""
    progress: int = 0  # 0-100


# Global ingestion status (in-memory)
_ingest_status = IngestStatus(status="idle")


@router.get("/files", response_model=List[FileInfo])
async def list_files():
    """List all files in the knowledge base.
    
    Returns flat list of files with metadata.
    """
    if not KNOWLEDGE_BASE_DIR.exists():
        return []
    
    files = []
    for txt_file in KNOWLEDGE_BASE_DIR.rglob("*.txt"):
        relative_path = txt_file.relative_to(KNOWLEDGE_BASE_DIR)
        category = str(relative_path.parent) if relative_path.parent != Path(".") else "root"
        
        files.append(FileInfo(
            path=str(relative_path),
            category=category,
            filename=txt_file.name,
            size=txt_file.stat().st_size
        ))
    
    return files


@router.get("/tree", response_model=FileTreeNode)
async def get_file_tree():
    """Get hierarchical file tree for knowledge base.
    
    Returns tree structure for file browser UI.
    """
    if not KNOWLEDGE_BASE_DIR.exists():
        return FileTreeNode(
            name="knowledge_base",
            type="directory",
            path="",
            children=[]
        )
    
    def build_tree(directory: Path) -> FileTreeNode:
        """Recursively build file tree."""
        node = FileTreeNode(
            name=directory.name,
            type="directory",
            path=str(directory.relative_to(KNOWLEDGE_BASE_DIR)) if directory != KNOWLEDGE_BASE_DIR else ""
        )
        
        # Add subdirectories
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir():
                node.children.append(build_tree(subdir))
        
        # Add files
        for file in sorted(directory.glob("*.txt")):
            node.children.append(FileTreeNode(
                name=file.name,
                type="file",
                path=str(file.relative_to(KNOWLEDGE_BASE_DIR)),
                size=file.stat().st_size
            ))
        
        return node
    
    return build_tree(KNOWLEDGE_BASE_DIR)


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    category: str = Form(None)
):
    """Upload one or more text files to knowledge base.
    
    Files are saved preserving their folder structure from the client.
    If webkitRelativePath is provided in filename (from folder upload),
    it preserves the directory structure. Otherwise saves to category/ or root.
    """
    uploaded = []
    errors = []
    
    for file in files:
        # Validate file type
        if not file.filename.endswith(".txt"):
            errors.append(f"{file.filename}: Only .txt files allowed")
            continue
        
        # Handle folder structure preservation
        # When uploading folders, browser sends webkitRelativePath in filename
        # Format is usually "foldername/subfolder/file.txt"
        filename = file.filename
        
        # Check if filename contains path separators (folder upload)
        if "/" in filename or "\\" in filename:
            # Preserve folder structure
            # Remove leading folder name and use rest as path
            parts = filename.replace("\\", "/").split("/")
            if len(parts) > 1:
                # Skip the root folder name, preserve subdirectories
                relative_path = "/".join(parts[1:]) if len(parts) > 2 else parts[-1]
                file_path = KNOWLEDGE_BASE_DIR / relative_path.replace("/", os.sep)
            else:
                file_path = KNOWLEDGE_BASE_DIR / parts[-1]
        elif category:
            # Use category if specified
            file_path = KNOWLEDGE_BASE_DIR / category / filename
        else:
            # Save to root
            file_path = KNOWLEDGE_BASE_DIR / filename
        
        # Create directory structure
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        try:
            content = await file.read()
            file_path.write_bytes(content)
            uploaded.append(str(file_path.relative_to(KNOWLEDGE_BASE_DIR)))
            logger.info(f"Uploaded file: {file_path}")
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            logger.error(f"Failed to upload {file.filename}: {e}")
    
    return {
        "uploaded": uploaded,
        "errors": errors,
        "count": len(uploaded)
    }


@router.delete("/files/{file_path:path}")
async def delete_file(file_path: str):
    """Delete a file from the knowledge base.
    
    Args:
        file_path: Relative path from knowledge_base/ directory
    """
    full_path = KNOWLEDGE_BASE_DIR / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    try:
        full_path.unlink()
        logger.info(f"Deleted file: {full_path}")
        return {"message": f"Deleted {file_path}"}
    except Exception as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def trigger_ingest(background_tasks: BackgroundTasks):
    """Trigger ingestion of knowledge base files.
    
    Runs in background. Use /status or WebSocket for progress.
    """
    global _ingest_status
    
    if _ingest_status.status == "running":
        raise HTTPException(status_code=409, detail="Ingestion already in progress")
    
    def run_ingestion():
        """Background task to run ingestion."""
        global _ingest_status
        _ingest_status = IngestStatus(status="running", message="Starting ingestion...")
        
        try:
            from api.config_manager import ConfigManager
            config_manager = ConfigManager()
            settings = config_manager.get_default_config()
            
            _ingest_status.message = "Loading documents..."
            _ingest_status.progress = 30
            
            ingest_knowledge_base(settings)
            
            _ingest_status = IngestStatus(
                status="completed",
                message="Ingestion completed successfully",
                progress=100
            )
            logger.info("Ingestion completed successfully")
        except Exception as e:
            _ingest_status = IngestStatus(
                status="error",
                message=f"Ingestion failed: {str(e)}",
                progress=0
            )
            logger.error(f"Ingestion failed: {e}")
    
    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion started", "status": "running"}


@router.get("/ingest/status", response_model=IngestStatus)
async def get_ingest_status():
    """Get current status of ingestion operation."""
    return _ingest_status


@router.websocket("/ws/ingest")
async def ingest_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time ingestion progress.
    
    Sends periodic updates while ingestion is running.
    """
    await websocket.accept()
    
    try:
        while True:
            # Send current status
            await websocket.send_json(_ingest_status.model_dump())
            
            # If completed or error, close connection
            if _ingest_status.status in ["completed", "error"]:
                break
            
            # Wait before next update
            import asyncio
            await asyncio.sleep(0.5)
        
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("Ingestion WebSocket disconnected")
