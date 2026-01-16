"""FastAPI backend for RAG Evaluation Framework Web UI.

Architecture:
- Thin wrapper around existing rag_app/ and eval/ modules
- Configuration overrides work like CLI parameters (per-request)
- WebSocket endpoints for real-time progress during long operations
- File uploads handled with proper validation
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config_manager import ConfigManager
from api.routers import config, datasets, eval_router, knowledge_base, query, system, prompts

# Load .env from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env", override=True)

config_manager = ConfigManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    config_manager.load_default_config()
    
    # Initialize prompt manager (creates default prompts if needed)
    from rag_app.prompt_manager import get_prompt_manager
    get_prompt_manager()
    
    yield

app = FastAPI(
    title="RAG Evaluation Framework API",
    description="Web API for interactive RAG testing and evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config.router, prefix="/api/config", tags=["Configuration"])
app.include_router(knowledge_base.router, prefix="/api/knowledge-base", tags=["Knowledge Base"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(eval_router.router, prefix="/api/eval", tags=["Evaluation"])
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(prompts.router, prefix="/api/prompts", tags=["Prompts"])

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


@app.get("/")
async def root():
    """Root endpoint redirecting to API docs."""
    return {"message": "RAG Evaluation Framework API. Visit /docs for API documentation."}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag-evaluation-api",
        "version": "1.0.0"
    }

app.state.config_manager = config_manager
