from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .embeddings import HashEmbedding
from .factory import make_backend
from .indexer import RepositoryIndexer
from .retrieval import Retriever

app = typer.Typer(help="ForgeContext grounded developer context CLI")
context_app = typer.Typer(help="Repository context operations")
app.add_typer(context_app, name="context")
console = Console()


def _services():
    settings = Settings.from_env()
    backend = make_backend(settings)
    embedder = HashEmbedding(settings.embedding_dimensions)
    return settings, backend, embedder


@context_app.command("sync")
def context_sync(path: Path = typer.Argument(Path("."), exists=True, file_okay=False)) -> None:
    """Index a repository into the configured context store."""
    settings, backend, embedder = _services()
    report = RepositoryIndexer(backend, settings, embedder).sync(path)
    console.print(f"[bold green]Indexed {report.chunks_indexed} chunks[/bold green] from {report.files_indexed} files")
    for language, count in sorted(report.languages.items()):
        console.print(f"  {language}: {count}")


@context_app.command("status")
def context_status() -> None:
    """Show current local/remote index status."""
    settings, backend, _ = _services()
    mode = "Qdrant" if settings.qdrant_url else f"local ({settings.state_dir})"
    console.print(f"Backend: [bold]{mode}[/bold]")
    console.print(f"Chunks: [bold]{backend.count()}[/bold]")


@app.command("ask")
def ask(question: str, limit: int = typer.Option(6, min=1, max=20)) -> None:
    """Retrieve grounded repository context for a natural-language question."""
    _, backend, embedder = _services()
    bundle = Retriever(backend, embedder).ask(question, limit=limit)
    if not bundle.hits:
        console.print("No context found. Run `dev-ai context sync .` first.")
        raise typer.Exit(code=1)

    table = Table(title="Grounded context")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Symbol")
    table.add_column("Excerpt")
    for hit in bundle.hits:
        src = hit.chunk.source
        excerpt = hit.chunk.text.replace("\n", " ")[:180]
        table.add_row(
            f"{hit.final_score:.3f}",
            f"{src.path}:{src.start_line}-{src.end_line}",
            src.symbol or "—",
            excerpt,
        )
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


if __name__ == "__main__":
    app()
