"""
simple-embed CLI — all commands communicate with the FastAPI backend via HTTP,
so the CLI can be used from outside a Docker container.

Usage examples
--------------
# Upload a single file
python -m cli.main upload report.pdf

# Upload multiple files
python -m cli.main upload doc1.pdf doc2.docx notes.md

# Upload every supported file in a folder (recursive)
python -m cli.main upload --folder ./documents

# Filter by extension and preview without actually uploading
python -m cli.main upload --folder ./docs --ext pdf,docx --dry-run

# Concurrent uploads with 4 workers
python -m cli.main upload --folder ./docs --workers 4

# Search
python -m cli.main search "chi phí vận hành"
python -m cli.main search "revenue growth" --mode semantic --top-k 20
python -m cli.main search "(?i)doanh.thu" --mode regex

# List indexed documents
python -m cli.main list

# Delete a document
python -m cli.main delete <doc_id>

# Check service health
python -m cli.main health

Environment variables
---------------------
EMBED_API_URL   Base URL of the API server (default: http://localhost:8000)
"""

import concurrent.futures
import sys
from pathlib import Path

import click
import httpx

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".txt", ".md", ".csv"}
)

_DEFAULT_API_URL = "http://localhost:8000"


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--api-url",
    default=_DEFAULT_API_URL,
    envvar="EMBED_API_URL",
    show_default=True,
    help="Base URL of the simple-embed API server.",
)
@click.pass_context
def cli(ctx: click.Context, api_url: str) -> None:
    """simple-embed: semantic document search CLI."""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api_url.rstrip("/")


# ── upload ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--folder",
    "-f",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Recursively upload all supported files from FOLDER.",
)
@click.option(
    "--ext",
    default=None,
    help="Comma-separated extensions to include when using --folder (e.g. pdf,docx).",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    show_default=True,
    type=click.IntRange(1, 16),
    help="Number of concurrent upload threads.",
)
@click.option("--dry-run", is_flag=True, help="List files that would be uploaded without uploading.")
@click.pass_context
def upload(
    ctx: click.Context,
    files: tuple[Path, ...],
    folder: str | None,
    ext: str | None,
    workers: int,
    dry_run: bool,
) -> None:
    """Upload one or more documents for indexing."""
    api_url: str = ctx.obj["api_url"]

    # ── Collect all target file paths ──────────────────────────────────────
    allowed_exts: frozenset[str]
    if ext:
        allowed_exts = frozenset(
            "." + e.strip().lstrip(".").lower() for e in ext.split(",") if e.strip()
        )
    else:
        allowed_exts = SUPPORTED_EXTENSIONS

    targets: list[Path] = []
    for p in files:
        if p.is_dir():
            found = sorted(
                item for item in p.rglob("*")
                if item.is_file() and item.suffix.lower() in allowed_exts
            )
            targets.extend(found)
        elif p.is_file():
            if p.suffix.lower() in allowed_exts:
                targets.append(p)
            else:
                click.echo(
                    f"  {click.style(p.name, bold=True)}  "
                    f"{click.style('SKIP', fg='yellow')} (unsupported extension)",
                    err=True,
                )

    if folder:
        folder_path = Path(folder)
        found = sorted(
            f for f in folder_path.rglob("*")
            if f.is_file() and f.suffix.lower() in allowed_exts
        )
        targets.extend(found)

    if not targets:
        click.echo("No files found to upload.")
        return

    click.echo(f"Found {len(targets)} file(s) to upload.")

    if dry_run:
        for t in targets:
            click.echo(f"  [dry-run] {t}")
        return

    # ── Upload with optional concurrency ─────────────────────────────────
    if workers == 1:
        for path in targets:
            _upload_one(api_url, path)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_upload_one, api_url, p): p for p in targets}
            for fut in concurrent.futures.as_completed(futures):
                # Exceptions are already handled inside _upload_one
                fut.result()


def _upload_one(api_url: str, path: Path) -> None:
    label = click.style(path.name, bold=True)
    try:
        with open(path, "rb") as fh:
            resp = httpx.post(
                f"{api_url}/documents/upload",
                files={"file": (path.name, fh, _mime(path))},
                timeout=300.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "?")
            if status == "duplicate":
                click.echo(f"  {label}  {click.style('SKIP', fg='yellow')} (already indexed)")
            elif status == "empty":
                click.echo(f"  {label}  {click.style('EMPTY', fg='yellow')} (no text extracted)")
            else:
                chunks = data.get("chunks", 0)
                pages = data.get("pages", 0)
                click.echo(
                    f"  {label}  {click.style('OK', fg='green')} "
                    f"({pages} page(s), {chunks} chunk(s))"
                )
        else:
            detail = _parse_error(resp)
            click.echo(
                f"  {label}  {click.style('FAIL', fg='red')} "
                f"[{resp.status_code}] {detail}",
                err=True,
            )
    except httpx.ConnectError:
        click.echo(
            f"  {label}  {click.style('ERROR', fg='red')} "
            f"Cannot connect to {api_url}",
            err=True,
        )
    except Exception as exc:
        click.echo(
            f"  {label}  {click.style('ERROR', fg='red')} {exc}",
            err=True,
        )


def _mime(path: Path) -> str:
    _map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
    }
    return _map.get(path.suffix.lower(), "application/octet-stream")


# ── search ────────────────────────────────────────────────────────────────────

@cli.command(name="search")
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    default=10,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Maximum results per search layer.",
)
@click.option(
    "--mode",
    "-m",
    default="all",
    show_default=True,
    type=click.Choice(["all", "exact", "regex", "semantic"], case_sensitive=False),
    help="Search strategy.",
)
@click.option(
    "--show-text/--no-text",
    default=True,
    help="Show the matched chunk text in results.",
)
@click.pass_context
def search_cmd(
    ctx: click.Context,
    query: str,
    top_k: int,
    mode: str,
    show_text: bool,
) -> None:
    """Search indexed documents by query string."""
    api_url: str = ctx.obj["api_url"]

    try:
        resp = httpx.get(
            f"{api_url}/search/",
            params={"query": query, "top_k": top_k, "mode": mode},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        click.echo(f"ERROR: Cannot connect to {api_url}", err=True)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        click.echo(f"ERROR: {exc.response.status_code} — {_parse_error(exc.response)}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    merged = data.get("merged", [])
    total = data.get("total", 0)

    if total == 0:
        click.echo(f'No results for: "{query}"')
        return

    click.echo(
        f'\n{click.style(str(total), bold=True)} result(s) for '
        f'{click.style(repr(query), fg="cyan")}  '
        f'[mode={mode}]\n'
    )

    for idx, item in enumerate(merged, 1):
        meta = item.get("metadata", {})
        score = item.get("score", 0.0)
        source = item.get("source", "?")
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page_number", 0)
        doc_id = meta.get("doc_id", "")
        chunk_idx = meta.get("chunk_index", "?")

        source_color = {"exact": "green", "regex": "yellow", "semantic": "blue"}.get(source, "white")

        header = (
            f"[{idx}] {click.style(file_name, bold=True)}"
            f"  page={page}  chunk={chunk_idx}"
            f"  score={score:.3f}"
            f"  [{click.style(source, fg=source_color)}]"
        )
        click.echo(header)

        if show_text:
            text_preview = item.get("text", "")[:300].replace("\n", " ")
            click.echo(f"    {click.style(text_preview, fg='white', dim=True)}")

        if doc_id:
            file_url = f"{api_url}/documents/{doc_id}/file"
            click.echo(f"    {click.style(file_url, fg='cyan', underline=True)}")

        click.echo()


# ── list ──────────────────────────────────────────────────────────────────────

@cli.command(name="list")
@click.option("--limit", default=50, show_default=True, help="Max documents to show.")
@click.pass_context
def list_docs(ctx: click.Context, limit: int) -> None:
    """List all indexed documents."""
    api_url: str = ctx.obj["api_url"]

    try:
        resp = httpx.get(f"{api_url}/documents/", params={"limit": limit}, timeout=15.0)
        resp.raise_for_status()
        docs = resp.json()
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    if not docs:
        click.echo("No documents indexed yet.")
        return

    click.echo(f"\n{len(docs)} document(s):\n")
    for doc in docs:
        click.echo(
            f"  {click.style(doc.get('doc_id', '')[:8], fg='yellow')}…  "
            f"{click.style(doc.get('file_name', '?'), bold=True)}  "
            f"({doc.get('file_type', '?')}  "
            f"chunks={doc.get('total_chunks', '?')}  "
            f"page={doc.get('page_count', '?')}  "
            f"uploaded={doc.get('upload_ts', '?')[:10]})"
        )
    click.echo()


# ── delete ────────────────────────────────────────────────────────────────────

@cli.command(name="delete")
@click.argument("doc_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def delete_doc(ctx: click.Context, doc_id: str, yes: bool) -> None:
    """Delete a document by its doc_id."""
    api_url: str = ctx.obj["api_url"]

    if not yes:
        click.confirm(f"Delete document '{doc_id}'?", abort=True)

    try:
        resp = httpx.delete(f"{api_url}/documents/{doc_id}", timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        click.echo(
            f"Deleted {click.style(doc_id, bold=True)} "
            f"({data.get('chunks_removed', 0)} chunk(s) removed)."
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            click.echo(f"Document '{doc_id}' not found.", err=True)
        else:
            click.echo(f"ERROR: {exc.response.status_code} — {_parse_error(exc.response)}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)


# ── health ────────────────────────────────────────────────────────────────────

@cli.command(name="health")
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check the health of the API server."""
    api_url: str = ctx.obj["api_url"]

    try:
        resp = httpx.get(f"{api_url}/health", timeout=5.0)
        data = resp.json()
    except Exception as exc:
        click.echo(f"ERROR: Cannot reach {api_url} — {exc}", err=True)
        sys.exit(1)

    status = data.get("status", "unknown")
    color = "green" if status == "ok" else "red"
    click.echo(f"Status: {click.style(status.upper(), fg=color, bold=True)}")
    click.echo(f"  Ollama:  {'✓' if data.get('ollama_reachable') else '✗'}")
    click.echo(f"  ChromaDB: {'✓' if data.get('chroma_reachable') else '✗'}")
    click.echo(f"  Version: {data.get('version', '?')}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        return body.get("detail") or body.get("error") or resp.text[:200]
    except Exception:
        return resp.text[:200]


if __name__ == "__main__":
    cli()
