"""RAG query and answer generation."""

import logging
import time
from datetime import datetime
from typing import Any, Dict

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole

from rag_app.config import Settings as AppSettings
from rag_app.index import get_index
from rag_app.prompts import PROMPT_VERSION, format_context, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from rag_app.prompt_manager import get_prompt_manager
from rag_app.retrievers import VectorRetriever
from rag_app.utils import truncate_text, configure_llama_index

logger = logging.getLogger(__name__)


def answer_question(question: str, settings: AppSettings, top_k: int = None) -> Dict[str, Any]:
    """Answer a question using RAG pipeline."""
    # Configure LlamaIndex with settings
    configure_llama_index(settings)
    
    top_k = top_k or settings.top_k
    
    index = get_index(settings)
    retriever = VectorRetriever(index, settings)
    
    # Time retrieval stage
    retrieval_start = time.perf_counter()
    nodes = retriever.retrieve(question, top_k)
    retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
    
    if not nodes:
        return {
            "question": question,
            "answer": "I couldn't find any relevant information to answer this question.",
            "contexts": [],
            "sources": [],
            "prompt_version": PROMPT_VERSION,
            "config_snapshot": _get_config_snapshot(settings, top_k),
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": 0.0,
            "total_time_ms": retrieval_time_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    
    context_str = format_context(nodes)
    user_prompt = USER_PROMPT_TEMPLATE.format(context_str=context_str, query_str=question)
    
    # Load system prompt from library or fallback to default
    prompt_manager = get_prompt_manager()
    system_prompt = prompt_manager.get_prompt("rag", settings.rag_system_prompt_title)
    if not system_prompt:
        logger.warning(f"Prompt '{settings.rag_system_prompt_title}' not found, using default")
        system_prompt = SYSTEM_PROMPT
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    
    # Time generation stage
    generation_start = time.perf_counter()
    response = Settings.llm.chat(messages)
    answer = response.message.content
    generation_time_ms = (time.perf_counter() - generation_start) * 1000
    
    total_time_ms = retrieval_time_ms + generation_time_ms
    
    # Extract token usage if available
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    try:
        if hasattr(response, 'raw') and hasattr(response.raw, 'usage'):
            usage = response.raw.usage
            prompt_tokens = getattr(usage, 'prompt_tokens', None)
            completion_tokens = getattr(usage, 'completion_tokens', None)
            total_tokens = getattr(usage, 'total_tokens', None)
    except Exception as e:
        logger.debug(f"Could not extract token usage: {e}")
    
    sources = [
        {
            "source_path": node.metadata.get("source_path", "unknown"),
            "category": node.metadata.get("category", "unknown"),
            "rank": rank,
            "score": node.score,
            "snippet": truncate_text(node.text, 200),
        }
        for rank, node in enumerate(nodes, 1)
    ]
    
    return {
        "question": question,
        "answer": answer,
        "contexts": [node.text for node in nodes],
        "sources": sources,
        "prompt_version": PROMPT_VERSION,
        "config_snapshot": _get_config_snapshot(settings, top_k),
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_time_ms": total_time_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _get_config_snapshot(settings: AppSettings, top_k: int) -> Dict[str, Any]:
    """Create complete configuration snapshot from all Settings fields."""
    snapshot = settings.model_dump()
    snapshot["top_k"] = top_k
    snapshot["prompt_version"] = PROMPT_VERSION
    snapshot["timestamp_utc"] = datetime.utcnow().isoformat()
    return snapshot
