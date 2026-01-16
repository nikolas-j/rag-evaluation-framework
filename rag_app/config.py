"""Configuration management for RAG Evaluation Framework."""

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
load_dotenv(_env_file, override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and YAML config."""

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # OpenAI API
    openai_api_key: str = Field(..., description="OpenAI API key (required)")

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model for generation"
    )
    
    # Judge Configuration
    judge_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for LLM-as-judge evaluation"
    )
    judge_num_samples: int = Field(
        default=1,
        ge=1,
        description="Number of judge samples to average per evaluation"
    )
    judge_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM judge"
    )

    # Vector Store
    vector_store_dir: str = Field(
        default="./storage/chroma",
        description="ChromaDB vector store directory"
    )
    collection_name: str = Field(
        default="knowledge_base",
        description="ChromaDB collection name"
    )

    # Retrieval Configuration
    retrieval_strategy: str = Field(
        default="vector",
        description="Retrieval strategy: 'vector', 'keyword', 'hybrid', or 'rerank'"
    )
    top_k: int = Field(default=5, description="Number of top documents to retrieve")
    chunk_size: int = Field(default=512, description="Text chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")

    # Overall Score Weights
    overall_score_weights: Dict[str, float] = Field(
        default={},
        description="Weights for computing overall score"
    )
    
    # Token Pricing (per million tokens)
    input_token_price_per_million: float = Field(
        default=0.150,
        ge=0.0,
        description="Price per million input tokens (USD)"
    )
    output_token_price_per_million: float = Field(
        default=0.600,
        ge=0.0,
        description="Price per million output tokens (USD)"
    )
    
    # Prompt Selection
    rag_system_prompt_title: str = Field(
        default="Default v1.0",
        description="Title of RAG system prompt to use from prompt library"
    )
    eval_prompt_contextual_precision: str = Field(
        default="Default v1.0",
        description="Prompt title for contextual_precision metric"
    )
    eval_prompt_correctness: str = Field(
        default="Default v1.0",
        description="Prompt title for correctness metric"
    )
    eval_prompt_contextual_relevance: str = Field(
        default="Default v1.0",
        description="Prompt title for contextual_relevance metric"
    )
    eval_prompt_faithfulness: str = Field(
        default="Default v1.0",
        description="Prompt title for faithfulness metric"
    )

    # Evaluation Performance Settings
    include_metric_reasons: bool = Field(
        default=False,
        description="Include detailed explanations for metrics (slower)"
    )
    max_contexts_for_eval: int = Field(
        default=3,
        ge=1,
        description="Maximum contexts to pass to eval metrics (null = all)"
    )

    def validate_api_key(self) -> bool:
        """Check if OpenAI API key is set."""
        return bool(self.openai_api_key and self.openai_api_key != "sk-your-api-key-here")
    
    def get_eval_prompt_titles(self) -> Dict[str, str]:
        """Get eval prompt titles as dict for backward compatibility."""
        return {
            "contextual_precision": self.eval_prompt_contextual_precision,
            "correctness": self.eval_prompt_correctness,
            "contextual_relevance": self.eval_prompt_contextual_relevance,
            "faithfulness": self.eval_prompt_faithfulness,
        }


def get_settings() -> Settings:
    """Get application settings.
    
    Creates a new Settings instance which loads from .env and uses Field defaults.
    """
    return Settings()