"""Command-line interface for RAG evaluation framework."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from eval.runner import run_eval
from rag_app.config import get_settings
from rag_app.ingest import ingest_knowledge_base
from rag_app.rag import answer_question

app = typer.Typer(help="CLI for RAG evaluation framework")
console = Console()
logger = logging.getLogger(__name__)


def check_api_key(settings) -> bool:
    if not settings.validate_api_key():
        console.print(Panel(
            "[bold red]OpenAI API key not found[/bold red]\nSet OPENAI_API_KEY in .env file.",
            title="Configuration Error",
            border_style="red",
        ))
        return False
    return True


@app.command()
def ingest():
    """Ingest documents from knowledge_base/ into vector store."""
    console.print("\n[bold cyan]Ingesting Knowledge Base[/bold cyan]\n")
    
    try:
        settings = get_settings()
        
        if not check_api_key(settings):
            raise typer.Exit(code=1)
        
        kb_path = Path("knowledge_base")
        if not kb_path.exists():
            console.print(Panel(
                "[bold red]knowledge_base/ directory not found[/bold red]",
                border_style="red",
            ))
            raise typer.Exit(code=1)
        
        txt_files = list(kb_path.rglob("*.txt"))
        if not txt_files:
            console.print(Panel(
                "[bold yellow]No .txt files found in knowledge_base/[/bold yellow]",
                border_style="yellow",
            ))
            raise typer.Exit(code=1)
        
        console.print(f"Documents found: {len(txt_files)}")
        console.print(f"Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}\n")
        
        ingest_knowledge_base(settings)
        
        console.print("\n[bold green]✓ Ingestion complete[/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        logger.error("Ingestion failed", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Question to answer"),
):
    """Answer a question using the RAG pipeline."""
    console.print("\n[bold cyan]RAG Query[/bold cyan]\n")
    
    try:
        settings = get_settings()
        
        if not check_api_key(settings):
            raise typer.Exit(code=1)
        
        console.print(f"[bold]Question:[/bold] {question}\n")
        
        with console.status("[cyan]Processing...[/cyan]", spinner="dots"):
            result = answer_question(question, settings)
        
        console.print("[bold green]Answer:[/bold green]")
        console.print(Panel(result["answer"], border_style="green"))
        
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(result["sources"], 1):
            console.print(f"{i}. [cyan]{source['source_path']}[/cyan]")
        
        console.print(f"\n[dim]Retrieved: {len(result['contexts'])} contexts[/dim]")
        console.print(f"[dim]Tokens: {result.get('total_tokens', 'N/A')}[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        logger.error("Query failed", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    dataset: str = typer.Option(
        "QA_testing_sets/golden.json",
        "--dataset",
        "-d",
        help="Path to test dataset JSON file"
    ),
    num_questions: int = typer.Option(
        None,
        "--num-questions",
        "-n",
        help="Limit to first N questions"
    ),
    run_name: str = typer.Option(
        None,
        "--run-name",
        "-r",
        help="Custom name for evaluation run"
    ),
):
    """Run evaluation on test dataset with quality metrics."""
    console.print("\n[bold cyan]Running Evaluation[/bold cyan]\n")
    
    try:
        settings = get_settings()
        
        if not check_api_key(settings):
            raise typer.Exit(code=1)
        
        if not Path(dataset).exists():
            console.print(f"\n[bold red]Dataset not found: {dataset}[/bold red]\n")
            raise typer.Exit(code=1)
        
        run_folder = run_eval(dataset, settings, run_name, num_questions)
        
        console.print(f"\n[bold green]✓ Evaluation complete[/bold green]")
        console.print(f"Results: [cyan]{run_folder}[/cyan]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        logger.error("Evaluation failed", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def reset():
    """Delete all vector store data."""
    console.print("\n[bold yellow]⚠ Vector Store Reset[/bold yellow]\n")
    
    try:
        settings = get_settings()
        
        import chromadb
        
        persist_dir = Path(settings.vector_store_dir)
        
        if not persist_dir.exists():
            console.print("[yellow]Vector store doesn't exist[/yellow]\n")
            return
        
        confirm = typer.confirm(
            f"Delete all data in {persist_dir}?",
            default=False
        )
        
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]\n")
            return
        
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            chroma_client.delete_collection(name=settings.collection_name)
            console.print(f"[green]✓ Deleted collection: {settings.collection_name}[/green]")
        except Exception:
            pass
        
        console.print("\n[bold green]✓ Reset complete[/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        logger.error("Reset failed", exc_info=True)
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
