from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .dependency import analyze_impact, git_changed_files
from .evaluation import evaluate_retrieval, load_eval_cases
from .factory import make_backend, make_embedder
from .indexer import RepositoryIndexer
from .retrieval import Retriever

app = typer.Typer(help="ForgeContext grounded developer context CLI")
context_app = typer.Typer(help="Repository context operations")
eval_app = typer.Typer(help="Retrieval evaluation operations")
app.add_typer(context_app, name="context")
app.add_typer(eval_app, name="eval")
console = Console()


def _services():
    settings = Settings.from_env()
    embedder = make_embedder(settings)
    backend = make_backend(settings, dimensions=embedder.dimensions)
    return settings, backend, embedder


@context_app.command("sync")
def context_sync(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False),
    force: bool = typer.Option(False, "--force", help="Rebuild every file instead of incremental sync."),
) -> None:
    """Index a repository into the configured context store."""
    settings, backend, embedder = _services()
    report = RepositoryIndexer(backend, settings, embedder).sync(path, force=force)
    console.print(
        f"[bold green]Index contains {report.chunks_indexed} chunks[/bold green] "
        f"({report.added_files} added, {report.changed_files} changed, "
        f"{report.unchanged_files} unchanged, {report.deleted_files} deleted files)"
    )
    for language, count in sorted(report.languages.items()):
        console.print(f"  {language}: {count}")


@context_app.command("status")
def context_status() -> None:
    """Show current local/remote index status."""
    settings, backend, embedder = _services()
    mode = "Qdrant" if settings.qdrant_url else f"local ({settings.state_dir})"
    console.print(f"Backend: [bold]{mode}[/bold]")
    console.print(f"Chunks: [bold]{backend.count()}[/bold]")
    console.print(
        f"Embeddings: [bold]{settings.embedding_provider}[/bold] "
        f"({getattr(embedder, 'model_name', settings.embedding_model or 'built-in')}, {embedder.dimensions}d)"
    )


@app.command("ask")
def ask(
    question: str,
    limit: int = typer.Option(6, min=1, max=20),
    json_output: bool = typer.Option(False, "--json", help="Emit a structured grounded answer."),
) -> None:
    """Retrieve grounded repository context for a natural-language question."""
    _, backend, embedder = _services()
    retriever = Retriever(backend, embedder)
    grounded = retriever.grounded_answer(question, limit=limit)
    if not grounded.citations:
        console.print("No context found. Run `dev-ai context sync .` first.")
        raise typer.Exit(code=1)

    if json_output:
        console.print_json(json.dumps(grounded.model_dump(mode="json")))
        return

    console.print(f"[bold]{grounded.answer}[/bold]")
    table = Table(title=f"Evidence · confidence {grounded.confidence:.3f}")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Symbol")
    for citation in grounded.citations:
        table.add_row(f"{citation.score:.3f}", citation.pointer, citation.symbol or "—")
    console.print(table)


@app.command("explain")
def explain(target: str) -> None:
    """Show indexed context for a file path or path fragment."""
    _, backend, embedder = _services()
    bundle = Retriever(backend, embedder).ask(target, limit=8)
    matches = [hit for hit in bundle.hits if target.split(":", 1)[0].lower() in hit.chunk.source.path.lower()]
    for hit in matches or bundle.hits[:3]:
        src = hit.chunk.source
        console.rule(f"{src.path}:{src.start_line}-{src.end_line}")
        console.print(hit.chunk.text)


@app.command("impact")
def impact(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False),
    changed: list[str] | None = typer.Option(None, "--changed", help="Changed repository path; repeatable."),
    base: str = typer.Option("HEAD~1", help="Git diff base when --changed is omitted."),
    head: str = typer.Option("HEAD", help="Git diff head when --changed is omitted."),
    max_depth: int = typer.Option(4, min=0, max=20),
) -> None:
    """Show files that may be affected by changed code through local imports."""
    changed_files = changed or git_changed_files(path, base=base, head=head)
    report = analyze_impact(path, changed_files, max_depth=max_depth)
    table = Table(title="Dependency-aware impact")
    table.add_column("Depth", justify="right")
    table.add_column("File")
    table.add_column("Reason")
    for item in report.impacted_files:
        table.add_row(str(item.depth), item.path, item.reason)
    console.print(table)
    console.print(f"Edges considered: {report.edges_considered}")


@eval_app.command("run")
def eval_run(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    k: int = typer.Option(5, min=1, max=20),
) -> None:
    """Measure retrieval Hit@K and mean reciprocal rank against labeled questions."""
    _, backend, embedder = _services()
    cases = load_eval_cases(dataset)
    report = evaluate_retrieval(Retriever(backend, embedder), cases, k=k)
    console.print(f"Cases: [bold]{report.cases}[/bold]")
    console.print(f"Hit@{k}: [bold]{report.hit_rate_at_k:.3f}[/bold]")
    console.print(f"MRR: [bold]{report.mean_reciprocal_rank:.3f}[/bold]")
    failed = [result for result in report.results if not result.passed]
    if failed:
        console.print("[yellow]Missed queries:[/yellow]")
        for result in failed:
            console.print(f"  - {result.question}")


if __name__ == "__main__":
    app()
