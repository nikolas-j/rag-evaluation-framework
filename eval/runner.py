"""Evaluation runner - orchestrates RAG evaluation over dataset."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from eval.dataset_loader import load_dataset
from eval.metrics import compute_metrics, compute_overall_score
from eval.reporting import generate_summary
from rag_app.config import Settings as AppSettings
from rag_app.rag import answer_question
from rag_app.utils import truncate_text

logger = logging.getLogger(__name__)
console = Console()

MAX_CONTEXT_CHARS = 1500


def create_run_folder(run_name: str = None) -> Path:
    """Create run folder with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_folder = Path("storage/runs") / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def save_config_snapshot(run_folder: Path, settings: AppSettings) -> None:
    """Save complete configuration snapshot."""
    config_snapshot = settings.model_dump()
    config_snapshot["timestamp_utc"] = datetime.utcnow().isoformat()
    
    with open(run_folder / "config_snapshot.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)


def run_eval(
    dataset_path: str,
    settings: AppSettings,
    run_name: str = None,
    num_questions: int = None,
    selected_metrics: List[str] = None,
    progress_callback = None,
) -> Path:
    """Run evaluation on dataset and generate report.
    
    Args:
        progress_callback: Optional callback function to report progress.
                          Called with (current_question_num, total_questions, question_text)
    """
    console.print(f"\n[bold cyan]Starting Evaluation Run[/bold cyan]")
    console.print(f"Dataset: {dataset_path}")
    
    if selected_metrics:
        console.print(f"Metrics: {', '.join(selected_metrics)}")
    
    dataset = load_dataset(dataset_path)
    
    if num_questions and num_questions < len(dataset):
        dataset.records = dataset.records[:num_questions]
    
    console.print(f"Loaded {len(dataset)} evaluation records\n")
    
    run_folder = create_run_folder(run_name)
    console.print(f"Run folder: [green]{run_folder}[/green]\n")
    
    save_config_snapshot(run_folder, settings)
    
    metadata = {
        "run_id": str(uuid.uuid4()),
        "dataset_path": dataset_path,
        "dataset_name": dataset.name,
        "num_questions": len(dataset.records),
        "selected_metrics": selected_metrics or [],
        "run_name": run_name,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    
    with open(run_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    jsonl_file = run_folder / "report.jsonl"
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Evaluating {len(dataset)} questions...", total=len(dataset.records))
        
        for i, record in enumerate(dataset.records, 1):
            progress.update(task, description=f"[cyan]Evaluating {i}/{len(dataset)}: {record.question[:50]}...[/cyan]", advance=1)
            
            # Report progress to callback if provided
            if progress_callback:
                progress_callback(i, len(dataset.records), record.question)
            
            try:
                rag_result = answer_question(record.question, settings)
                
                contexts_for_eval = rag_result["contexts"]
                if settings.max_contexts_for_eval:
                    contexts_for_eval = contexts_for_eval[:settings.max_contexts_for_eval]
                
                metric_results = compute_metrics(
                    question=record.question,
                    expected_answer=record.expected_answer,
                    answer=rag_result["answer"],
                    contexts=contexts_for_eval,
                    settings=settings,
                    selected_metrics=selected_metrics,
                )
                
                overall_score = compute_overall_score(metric_results, settings.overall_score_weights)
                
                jsonl_record = {
                    "run_id": metadata["run_id"],
                    "record_id": record.id,
                    "question": record.question,
                    "expected_answer": record.expected_answer,
                    "expected_sources": record.expected_sources,
                    "answer": rag_result["answer"],
                    "contexts": [truncate_text(ctx, MAX_CONTEXT_CHARS) for ctx in rag_result["contexts"]],
                    "sources": rag_result["sources"],
                    "metrics": metric_results,
                    "overall_score": overall_score,
                    "config_snapshot": rag_result["config_snapshot"],
                    "retrieval_time_ms": rag_result.get("retrieval_time_ms", 0.0),
                    "generation_time_ms": rag_result.get("generation_time_ms", 0.0),
                    "total_time_ms": rag_result.get("total_time_ms", 0.0),
                    "prompt_tokens": rag_result.get("prompt_tokens"),
                    "completion_tokens": rag_result.get("completion_tokens"),
                    "total_tokens": rag_result.get("total_tokens"),
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
                
                with open(jsonl_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(jsonl_record) + "\n")
                
            except Exception as e:
                logger.error(f"Error processing record {record.id}: {e}", exc_info=True)
                console.print(f"[red]Error processing {record.id}: {e}[/red]")
    
    console.print("\n[bold green]Evaluation complete![/bold green]\n")
    console.print("[cyan]Generating summary report...[/cyan]\n")
    
    summary_dict, _ = generate_summary(run_folder, settings)
    
    console.print("[bold]" + "=" * 80 + "[/bold]")
    console.print("[bold cyan]EVALUATION SUMMARY[/bold cyan]")
    console.print("[bold]" + "=" * 80 + "[/bold]\n")
    console.print(f"[bold]Total Questions:[/bold] {summary_dict['total_questions']}\n")
    console.print("[bold]Metric Averages:[/bold]")
    for metric_name, avg_score in summary_dict["metric_averages"].items():
        console.print(f"  {metric_name.replace('_', ' ').title()}: {avg_score:.3f}")
    console.print(f"\n[bold]Average Overall Score:[/bold] {summary_dict['average_overall_score']:.3f}\n")
    console.print(f"[bold]Full report:[/bold] [green]{run_folder}[/green]\n")
    
    return run_folder
