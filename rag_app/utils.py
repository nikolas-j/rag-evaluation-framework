"""Utility functions for RAG application."""

import hashlib
from pathlib import Path

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def configure_llama_index(settings) -> None:
    """Configure LlamaIndex global settings."""
    Settings.llm = OpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap


def get_file_hash(file_path: Path) -> str:
    """Generate hash based on file content.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash of file content
    """
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def truncate_text(text: str, max_length: int = 1500) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_category_from_path(file_path: Path, knowledge_base_root: Path) -> str:
    """Extract category from file path."""
    relative_path = file_path.relative_to(knowledge_base_root)
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
