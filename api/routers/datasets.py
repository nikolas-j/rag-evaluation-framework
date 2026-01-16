"""Dataset management endpoints."""

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from eval.dataset_schema import EvaluationDataset

router = APIRouter()

DATASETS_DIR = Path("QA_testing_sets")


class DatasetInfo(BaseModel):
    """Information about a dataset file."""
    name: str
    path: str
    num_questions: int
    description: str = ""


@router.get("", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets."""
    if not DATASETS_DIR.exists():
        return []
    
    datasets = []
    for json_file in DATASETS_DIR.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate schema
            dataset = EvaluationDataset(**data)
            
            datasets.append(DatasetInfo(
                name=json_file.stem,
                path=str(json_file.relative_to(DATASETS_DIR)),
                num_questions=len(dataset.records),
                description=dataset.description or ""
            ))
        except Exception:
            # Skip invalid datasets
            continue
    
    return datasets


@router.get("/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get a specific dataset by name."""
    dataset_path = DATASETS_DIR / f"{dataset_name}.json"
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate schema
        dataset = EvaluationDataset(**data)
        
        return dataset.model_dump()
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset schema: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset JSON file."""
    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files allowed")
    
    # Read and parse JSON
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    
    # Validate schema
    try:
        dataset = EvaluationDataset(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset schema: {str(e)}")
    
    # Save file
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATASETS_DIR / file.filename
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return {
        "message": "Dataset uploaded successfully",
        "name": dataset_path.stem,
        "num_questions": len(dataset.records)
    }


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset file."""
    dataset_path = DATASETS_DIR / f"{dataset_name}.json"
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset_name == "golden":
        raise HTTPException(status_code=403, detail="Cannot delete default dataset")
    
    try:
        dataset_path.unlink()
        return {"message": f"Deleted dataset: {dataset_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_dataset(file: UploadFile = File(...)):
    """Validate a dataset without saving."""
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        dataset = EvaluationDataset(**data)
        
        return {
            "valid": True,
            "num_questions": len(dataset.records),
            "errors": []
        }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Invalid JSON: {str(e)}"]
        }
    except ValidationError as e:
        errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        return {
            "valid": False,
            "errors": errors
        }
