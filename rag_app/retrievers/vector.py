"""Vector similarity retrieval using LlamaIndex and ChromaDB."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from llama_index.core import VectorStoreIndex

from rag_app.config import Settings as AppSettings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedNode:
    """Standardized retrieval result."""
    text: str
    score: float
    metadata: Dict[str, Any]
    node_id: str


class VectorRetriever:
    """Dense vector retrieval using cosine similarity."""
    
    def __init__(self, index: VectorStoreIndex, settings: AppSettings):
        self.index = index
        self.settings = settings
    
    def retrieve(self, query: str, top_k: int) -> List[RetrievedNode]:
        """Retrieve top_k documents using vector similarity."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        return [
            RetrievedNode(
                text=node.node.get_content(),
                score=node.score if node.score is not None else 0.0,
                metadata=node.node.metadata or {},
                node_id=node.node.node_id,
            )
            for node in nodes
        ]
