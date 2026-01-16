"""Custom LLM-as-judge metrics registry and computation."""

import logging
from typing import Any, Dict, List, Callable

from rag_app.config import Settings as AppSettings
from .judges import (
    judge_contextual_precision,
    judge_contextual_relevance,
    judge_correctness,
    judge_faithfulness
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "contextual_precision",
    "contextual_relevance",
    "correctness",
    "faithfulness"
]

# Metric registry: add new metrics here with 1 line
METRIC_REGISTRY: Dict[str, Callable] = {
    "contextual_precision": judge_contextual_precision,
    "contextual_relevance": judge_contextual_relevance,
    "correctness": judge_correctness,
    "faithfulness": judge_faithfulness,
}


def compute_metrics(
    question: str,
    expected_answer: str,
    answer: str,
    contexts: List[str],
    settings: AppSettings,
    selected_metrics: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute selected evaluation metrics using LLM-as-judge."""
    selected_metrics = selected_metrics or DEFAULT_METRICS
    results = {}
    
    for name in selected_metrics:
        if name not in METRIC_REGISTRY:
            logger.warning(f"Unknown metric '{name}' - available: {list(METRIC_REGISTRY.keys())}")
            continue
        
        try:
            judge_func = METRIC_REGISTRY[name]
            
            # Call judge function - all parameters loaded from settings inside judge
            # Pass answer to all judges for consistency (even if not used by all)
            result = judge_func(
                question=question,
                contexts=contexts,
                expected_answer=expected_answer,
                answer=answer
            )
            
            # Check for judge errors
            if "error" in result:
                logger.error(f"Judge error for {name}: {result['error']}")
                results[name] = {
                    "score": 0.0,
                    "reason": f"Judge Error: {result['error']}",
                    "samples": []
                }
                continue
            
            results[name] = {
                "score": result["score"],
                "reason": result["verdict"],
                "samples": result.get("samples", [])
            }
            
        except Exception as e:
            logger.error(f"Error computing {name}: {e}")
            results[name] = {
                "score": 0.0,
                "reason": f"Error: {str(e)}",
                "samples": []
            }
    
    return results


def compute_overall_score(metric_results: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> float:
    """Compute weighted overall score."""
    if not weights:
        # No weights specified - use equal weighting for all metrics
        scores = [r["score"] for r in metric_results.values()]
        return sum(scores) / len(scores) if scores else 0.0
    
    weighted = [metric_results[m]["score"] * w for m, w in weights.items() if m in metric_results]
    total_weight = sum(w for m, w in weights.items() if m in metric_results)
    return sum(weighted) / total_weight if total_weight > 0 else 0.0
