"""Reporting utilities for evaluation results."""

import heapq
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag_app.config import Settings as AppSettings

logger = logging.getLogger(__name__)

TOP_N_WORST = 5
METRIC_N_WORST = 3


def generate_summary(run_folder: Path, settings: AppSettings) -> Tuple[Dict[str, Any], str]:
    """Generate summary statistics and markdown report."""
    results = []
    with open(run_folder / "report.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        raise ValueError("No results found")
    
    total_questions = len(results)
    
    # Get metrics actually used from first result
    first_result_metrics = results[0]["metrics"].keys() if results else []
    metric_names = list(first_result_metrics)
    
    metric_sums = {m: 0.0 for m in metric_names}
    
    for result in results:
        for metric_name in metric_names:
            score = result["metrics"].get(metric_name, {}).get("score", 0.0)
            metric_sums[metric_name] += score
    
    metric_averages = {m: metric_sums[m] / total_questions for m in metric_names}
    average_overall_score = sum(r["overall_score"] for r in results) / total_questions
    
    # Calculate average timing metrics
    average_retrieval_time = sum(r.get("retrieval_time_ms", 0.0) for r in results) / total_questions
    average_generation_time = sum(r.get("generation_time_ms", 0.0) for r in results) / total_questions
    average_total_time = sum(r.get("total_time_ms", 0.0) for r in results) / total_questions
    
    # Calculate average token usage (only from results with token data)
    results_with_tokens = [r for r in results if r.get("total_tokens") is not None]
    if results_with_tokens:
        average_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results_with_tokens) / len(results_with_tokens)
        average_completion_tokens = sum(r.get("completion_tokens", 0) for r in results_with_tokens) / len(results_with_tokens)
        average_total_tokens = sum(r.get("total_tokens", 0) for r in results_with_tokens) / len(results_with_tokens)
    else:
        average_prompt_tokens = None
        average_completion_tokens = None
        average_total_tokens = None
    
    worst_overall = heapq.nsmallest(TOP_N_WORST, results, key=lambda x: x["overall_score"])
    
    worst_by_metric = {}
    for metric_name in metric_names:
        worst = heapq.nsmallest(
            METRIC_N_WORST,
            results,
            key=lambda x: x["metrics"].get(metric_name, {}).get("score", 0.0)
        )
        worst_by_metric[metric_name] = [
            {
                "record_id": r["record_id"],
                "question": r["question"],
                "score": r["metrics"].get(metric_name, {}).get("score", 0.0),
                "answer": r["answer"][:200],
            }
            for r in worst
        ]
    
    summary_dict = {
        "total_questions": total_questions,
        "metric_averages": metric_averages,
        "average_overall_score": average_overall_score,
        "average_retrieval_time_ms": average_retrieval_time,
        "average_generation_time_ms": average_generation_time,
        "average_total_time_ms": average_total_time,
        "average_prompt_tokens": average_prompt_tokens,
        "average_completion_tokens": average_completion_tokens,
        "average_total_tokens": average_total_tokens,
        "worst_overall": [
            {
                "record_id": r["record_id"],
                "question": r["question"],
                "overall_score": r["overall_score"],
                "answer": r["answer"][:200],
            }
            for r in worst_overall
        ],
        "worst_by_metric": worst_by_metric,
    }
    
    markdown_text = _generate_markdown(summary_dict, settings, run_folder)
    
    with open(run_folder / "summary.md", "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    with open(run_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    
    return summary_dict, markdown_text


def _generate_markdown(summary: Dict[str, Any], settings: AppSettings, run_folder: Path) -> str:
    """Generate markdown formatted report."""
    lines = [
        "# RPerformance Metrics\n",
        f"- **Average Retrieval Time**: {summary.get('average_retrieval_time_ms', 0):.1f} ms",
        f"- **Average Generation Time**: {summary.get('average_generation_time_ms', 0):.1f} ms",
        f"- **Average Total Time**: {summary.get('average_total_time_ms', 0):.1f} ms\n",
    ]
    
    # Add token metrics if available
    if summary.get('average_total_tokens') is not None:
        lines.extend([
            "## Token Usage\n",
            f"- **Average Prompt Tokens**: {summary.get('average_prompt_tokens', 0):.0f}",
            f"- **Average Completion Tokens**: {summary.get('average_completion_tokens', 0):.0f}",
            f"- **Average Total Tokens**: {summary.get('average_total_tokens', 0):.0f}\n",
        ])
    
    lines.extend([
        "## AG Evaluation Summary",
        f"\n**Total Questions:** {summary['total_questions']}",
        f"**Average Overall Score:** {summary['average_overall_score']:.3f}\n",
        "## Metric Averages\n",
    ])
    
    for metric, avg in summary["metric_averages"].items():
        lines.append(f"- **{metric.replace('_', ' ').title()}**: {avg:.3f}")
    
    lines.append("\n## Worst Performing Questions (Overall)\n")
    for i, item in enumerate(summary["worst_overall"], 1):
        lines.append(f"### {i}. {item['question']}\n")
        lines.append(f"- **Overall Score**: {item['overall_score']:.3f}")
        lines.append(f"- **Answer**: {item['answer']}...\n")
    
    lines.append("## Worst by Metric\n")
    for metric, items in summary["worst_by_metric"].items():
        lines.append(f"### {metric.replace('_', ' ').title()}\n")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. **{item['question']}** - Score: {item['score']:.3f}\n")
    
    return "\n".join(lines)
