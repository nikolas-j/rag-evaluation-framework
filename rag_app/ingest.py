"""Document ingestion and vector store population using ChromaDB."""

import logging
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_app.config import Settings as AppSettings
from rag_app.utils import configure_llama_index, get_category_from_path, get_file_hash

logger = logging.getLogger(__name__)


def load_documents_from_directory(knowledge_base_path: str = "knowledge_base"):
    """Load all text documents from knowledge base directory."""
    knowledge_base_root = Path(knowledge_base_path)
    
    if not knowledge_base_root.exists():
        raise ValueError(f"Knowledge base directory not found: {knowledge_base_path}")
    
    txt_files = list(knowledge_base_root.rglob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {knowledge_base_path}")
    
    documents = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        category = get_category_from_path(file_path, knowledge_base_root)
        doc_id = get_file_hash(file_path)
        source_path = str(file_path.relative_to(knowledge_base_root))
        
        documents.append(Document(
            text=content,
            metadata={
                "source_path": source_path,
                "category": category,
                "doc_id": doc_id,
                "filename": file_path.name,
            },
            id_=doc_id,
        ))
    
    return documents


def ingest_knowledge_base(settings: AppSettings, clear_existing: bool = True) -> None:
    """Ingest documents from knowledge base into ChromaDB vector store.
    
    Args:
        settings: Application settings
        clear_existing: If True, delete existing collection before ingesting (default: True)
                       This ensures fresh ingestion without stale data.
    """
    configure_llama_index(settings)
    
    documents = load_documents_from_directory()
    logger.info(f"Loaded {len(documents)} documents from knowledge base")
    
    persist_dir = Path(settings.vector_store_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Clear existing collection if requested (default behavior)
    if clear_existing:
        try:
            chroma_client.delete_collection(name=settings.collection_name)
            logger.info(f"Deleted existing collection: {settings.collection_name}")
        except Exception as e:
            logger.info(f"No existing collection to delete or error: {e}")
    
    chroma_collection = chroma_client.get_or_create_collection(settings.collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    logger.info("Starting document indexing...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    
    storage_context.persist(persist_dir=str(persist_dir))
    logger.info(f"Successfully ingested {len(documents)} documents")