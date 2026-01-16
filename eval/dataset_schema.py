"""Dataset schema definitions for evaluation."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
    """A single evaluation dataset record."""
    id: str
    question: str
    expected_answer: str
    expected_sources: Optional[List[str]] = None


class EvaluationDataset(BaseModel):
    """Complete evaluation dataset."""
    name: str
    description: Optional[str] = None
    records: List[DatasetRecord] = Field(..., min_length=1)
    
    def __len__(self) -> int:
        return len(self.records)
