"""Dataset loading and validation."""

import json
from pathlib import Path

from eval.dataset_schema import EvaluationDataset


def load_dataset(dataset_path: str) -> EvaluationDataset:
    """Load and validate evaluation dataset from JSON."""
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return EvaluationDataset(**data)
