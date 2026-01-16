"""Prompt management system for versioned prompts."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PromptManager:
    """Manages versioned prompts for RAG and evaluation."""
    
    def __init__(self):
        """Initialize prompt manager and ensure storage exists."""
        PROMPTS_DIR.mkdir(exist_ok=True)
        self._rag_prompts_file = PROMPTS_DIR / "rag_prompts.json"
        self._eval_prompts_file = PROMPTS_DIR / "eval_prompts.json"
        
        # Initialize with defaults if files don't exist
        if not self._rag_prompts_file.exists():
            self._initialize_rag_defaults()
        if not self._eval_prompts_file.exists():
            self._initialize_eval_defaults()
    
    def _initialize_rag_defaults(self) -> None:
        """Create default RAG prompts."""
        from rag_app.prompts import SYSTEM_PROMPT
        
        defaults = [
            {
                "title": "Default v1.0",
                "content": SYSTEM_PROMPT,
                "type": "system",
                "created_at": datetime.utcnow().isoformat(),
                "description": "Baseline RAG system prompt - cite sources, stay within context"
            }
        ]
        
        with open(self._rag_prompts_file, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=2)
        
        logger.info(f"Initialized default RAG prompts: {self._rag_prompts_file}")
    
    def _initialize_eval_defaults(self) -> None:
        """Create default evaluation prompts."""
        from eval.prompts import (
            CONTEXTUAL_PRECISION_PROMPT,
            CORRECTNESS_PROMPT,
            FAITHFULNESS_PROMPT,
            CONTEXTUAL_RELEVANCE_PROMPT
        )
        
        defaults = [
            {
                "title": "Default v1.0",
                "metric": "contextual_precision",
                "content": CONTEXTUAL_PRECISION_PROMPT,
                "created_at": datetime.utcnow().isoformat(),
                "description": "Baseline contextual precision judge - evaluates relevance and ranking"
            },
            {
                "title": "Default v1.0",
                "metric": "contextual_relevance",
                "content": CONTEXTUAL_RELEVANCE_PROMPT,
                "created_at": datetime.utcnow().isoformat(),
                "description": "Baseline contextual relevance judge - evaluates if contexts contain necessary information"
            },
            {
                "title": "Default v1.0",
                "metric": "correctness",
                "content": CORRECTNESS_PROMPT,
                "created_at": datetime.utcnow().isoformat(),
                "description": "Baseline correctness judge - evaluates if answer matches expected answer"
            },
            {
                "title": "Default v1.0",
                "metric": "faithfulness",
                "content": FAITHFULNESS_PROMPT,
                "created_at": datetime.utcnow().isoformat(),
                "description": "Baseline faithfulness judge - evaluates if answer is grounded in contexts"
            }
        ]
        
        with open(self._eval_prompts_file, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=2)
        
        logger.info(f"Initialized default eval prompts: {self._eval_prompts_file}")
    
    def list_prompts(self, category: str) -> List[Dict]:
        """List all prompts for a category.
        
        Args:
            category: "rag" or "eval"
            
        Returns:
            List of prompt dictionaries
        """
        file_path = self._rag_prompts_file if category == "rag" else self._eval_prompts_file
        
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {category} prompts: {e}")
            return []
    
    def get_prompt(self, category: str, title: str, metric: Optional[str] = None) -> Optional[str]:
        """Get prompt content by title.
        
        Args:
            category: "rag" or "eval"
            title: Prompt title
            metric: For eval prompts, the metric name (e.g., "contextual_precision")
            
        Returns:
            Prompt content string or None if not found
        """
        prompts = self.list_prompts(category)
        
        for prompt in prompts:
            if category == "eval":
                # For eval, match both title and metric
                if prompt.get("title") == title and prompt.get("metric") == metric:
                    return prompt.get("content")
            else:
                # For RAG, match only title
                if prompt.get("title") == title:
                    return prompt.get("content")
        
        return None
    
    def save_prompt(
        self,
        category: str,
        title: str,
        content: str,
        description: str = "",
        metric: Optional[str] = None,
        prompt_type: str = "system"
    ) -> bool:
        """Save a new prompt or update existing.
        
        Args:
            category: "rag" or "eval"
            title: Prompt title (used for versioning)
            content: Prompt content
            description: Optional description
            metric: For eval prompts, the metric name
            prompt_type: "system" or "user"
            
        Returns:
            True if saved successfully
        """
        file_path = self._rag_prompts_file if category == "rag" else self._eval_prompts_file
        prompts = self.list_prompts(category)
        
        # Check if prompt with same title (and metric for eval) exists
        existing_index = None
        for i, p in enumerate(prompts):
            if category == "eval":
                if p.get("title") == title and p.get("metric") == metric:
                    existing_index = i
                    break
            else:
                if p.get("title") == title:
                    existing_index = i
                    break
        
        new_prompt = {
            "title": title,
            "content": content,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        if category == "rag":
            new_prompt["type"] = prompt_type
        else:
            new_prompt["metric"] = metric
        
        if existing_index is not None:
            # Update existing
            new_prompt["created_at"] = prompts[existing_index].get("created_at")
            new_prompt["modified_at"] = datetime.utcnow().isoformat()
            prompts[existing_index] = new_prompt
        else:
            # Add new
            prompts.append(new_prompt)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(prompts, f, indent=2)
            logger.info(f"Saved {category} prompt: {title}")
            return True
        except Exception as e:
            logger.error(f"Error saving {category} prompt: {e}")
            return False
    
    def delete_prompt(self, category: str, title: str, metric: Optional[str] = None) -> bool:
        """Delete a prompt.
        
        Args:
            category: "rag" or "eval"
            title: Prompt title
            metric: For eval prompts, the metric name
            
        Returns:
            True if deleted successfully
        """
        file_path = self._rag_prompts_file if category == "rag" else self._eval_prompts_file
        prompts = self.list_prompts(category)
        
        # Find and remove
        filtered = []
        for p in prompts:
            if category == "eval":
                if not (p.get("title") == title and p.get("metric") == metric):
                    filtered.append(p)
            else:
                if p.get("title") != title:
                    filtered.append(p)
        
        if len(filtered) == len(prompts):
            return False  # Nothing deleted
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2)
            logger.info(f"Deleted {category} prompt: {title}")
            return True
        except Exception as e:
            logger.error(f"Error deleting {category} prompt: {e}")
            return False


# Global instance
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
