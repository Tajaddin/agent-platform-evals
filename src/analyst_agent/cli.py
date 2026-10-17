"""Command line: serve the API, ask one question, show the schema prompt."""

from __future__ import annotations

import json

import typer

from analyst_agent.config import get_settings
from analyst_agent.logging import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:  # noqa: S104
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run(
        "analyst_agent.api:create_app",
        host=host,
        port=port,
        factory=True,
        workers=workers,
        timeout_graceful_shutdown=30,
    )


@app.command()
def ask(
    question: str,
    thread_id: str = "cli",
    approve: bool = typer.Option(False, help="auto-approve expensive queries"),
) -> None:
    """Ask one question through the full graph and print the response as JSON."""
    from analyst_agent.service import AgentService

    configure_logging(json_output=False)
    svc = AgentService(get_settings())
    resp = svc.chat(thread_id, question)
    if resp.status == "approval_required" and approve:
        resp = svc.approve(thread_id, True)
    typer.echo(json.dumps(resp.model_dump(), indent=2, default=str))
    svc.close()


@app.command()
def schema() -> None:
    """Print the schema description the analyst sees."""
    from analyst_agent.warehouse import Warehouse

    s = get_settings()
    wh = Warehouse(s.warehouse_path, s.allowed_schemas, s.row_limit, s.query_timeout_seconds)
    typer.echo(wh.schema_prompt())


if __name__ == "__main__":
    app()
