"""Index management and retrieval using ChromaDB."""

import logging
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_app.config import Settings as AppSettings
from rag_app.utils import configure_llama_index

logger = logging.getLogger(__name__)


def get_index(settings: AppSettings) -> VectorStoreIndex:
    """Load the vector store index from persistent storage."""
    configure_llama_index(settings)
    
    persist_dir = Path(settings.vector_store_dir)
    
    if not persist_dir.exists():
        raise ValueError(
            f"Vector store not found at {persist_dir}. "
            "Please run ingestion first: python -m scripts.cli ingest"
        )
    
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_collection(settings.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create index from existing vector store
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
