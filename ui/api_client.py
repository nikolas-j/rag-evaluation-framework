"""Centralized API client for backend communication."""

import os
import requests
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(override=True)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

class APIClient:
    @staticmethod
    def load_config() -> Dict[str, Any]:
        response = requests.get(f"{API_BASE_URL}/api/config")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def save_config(overrides: Dict[str, Any]) -> None:
        response = requests.post(
            f"{API_BASE_URL}/api/config/save",
            json={"overrides": overrides}
        )
        response.raise_for_status()
    
    @staticmethod
    def upload_kb_files(files) -> Dict[str, Any]:
        file_list = [("files", (f.name, f.getvalue(), "text/plain")) for f in files]
        response = requests.post(f"{API_BASE_URL}/api/knowledge-base/upload", files=file_list)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def trigger_ingestion() -> None:
        response = requests.post(f"{API_BASE_URL}/api/knowledge-base/ingest")
        response.raise_for_status()
    
    @staticmethod
    def get_ingestion_status() -> Dict[str, Any]:
        response = requests.get(f"{API_BASE_URL}/api/knowledge-base/ingest/status")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_kb_files() -> List[Dict[str, Any]]:
        response = requests.get(f"{API_BASE_URL}/api/knowledge-base/files")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_kb_tree() -> Dict[str, Any]:
        response = requests.get(f"{API_BASE_URL}/api/knowledge-base/tree")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def upload_test_set(file) -> Dict[str, Any]:
        files = {"file": (file.name, file.getvalue(), "application/json")}
        response = requests.post(f"{API_BASE_URL}/api/datasets/upload", files=files)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_datasets() -> List[Dict[str, Any]]:
        response = requests.get(f"{API_BASE_URL}/api/datasets")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def query_rag(question: str, config_overrides: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"question": question, "config_overrides": config_overrides}
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def start_evaluation(dataset: str, metrics: List[str], config_overrides: Dict[str, Any], run_name: Optional[str] = None, num_questions: Optional[int] = None) -> str:
        payload = {
            "dataset": f"QA_testing_sets/{dataset}.json",
            "config_overrides": config_overrides,
            "selected_metrics": metrics
        }
        if run_name:
            payload["run_name"] = run_name
        if num_questions:
            payload["num_questions"] = num_questions
        
        response = requests.post(f"{API_BASE_URL}/api/eval/run", json=payload)
        response.raise_for_status()
        return response.json().get("run_id")
    
    @staticmethod
    def get_run_details(run_id: str) -> Dict[str, Any]:
        response = requests.get(f"{API_BASE_URL}/api/eval/runs/{run_id}")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_runs() -> List[Dict[str, Any]]:
        response = requests.get(f"{API_BASE_URL}/api/eval/runs")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_available_metrics() -> List[str]:
        response = requests.get(f"{API_BASE_URL}/api/eval/available-metrics")
        response.raise_for_status()
        return response.json().get("metrics", [])
    
    @staticmethod
    def list_prompts(category: str) -> List[Dict[str, Any]]:
        response = requests.get(f"{API_BASE_URL}/api/prompts/{category}")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def save_prompt(category: str, title: str, content: str, description: str = "", prompt_type: str = "system") -> None:
        response = requests.post(
            f"{API_BASE_URL}/api/prompts/{category}",
            json={"title": title, "content": content, "description": description, "type": prompt_type}
        )
        response.raise_for_status()
    
    @staticmethod
    def save_eval_prompt(metric: str, title: str, content: str, description: str = "") -> None:
        response = requests.post(
            f"{API_BASE_URL}/api/prompts/eval",
            json={"title": title, "content": content, "description": description, "metric": metric}
        )
        response.raise_for_status()
    
    @staticmethod
    def delete_prompt(category: str, title: str, metric: Optional[str] = None) -> None:
        params = {"metric": metric} if metric else {}
        response = requests.delete(f"{API_BASE_URL}/api/prompts/{category}/{title}", params=params)
        response.raise_for_status()
    
    @staticmethod
    def reset_vector_store() -> Dict[str, Any]:
        """Reset/clear the vector store (deletes all indexed documents)."""
        response = requests.post(f"{API_BASE_URL}/api/system/reset")
        response.raise_for_status()
        return response.json()
