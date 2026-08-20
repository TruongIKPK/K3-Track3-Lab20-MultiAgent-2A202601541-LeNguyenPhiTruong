from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    run_benchmark,
    run_corpus_benchmark,
    run_multi_agent_workflow,
    run_single_agent_baseline,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline completion."""
    _init()
    _parse_query(query)
    state = run_single_agent_baseline(query)
    console.print(
        Panel.fit(state.final_answer or "No answer generated", title="Single-Agent Baseline")
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    _parse_query(query)
    state = run_multi_agent_workflow(query)
    console.print(result_panel(state))
    console.print(state.model_dump_json(indent=2))


def result_panel(state: ResearchState) -> Panel:
    return Panel(
        Markdown(state.final_answer or "No final answer"),
        title="Multi-Agent Synthesis Report",
        border_style="green",
    )


DEFAULT_CORPUS_DIR = (
    "src/ai_agent_offline_research_corpus_30_topics_v2/ai_agent_offline_research_corpus_v2/topics"
)


@app.command("benchmark")
def benchmark(
    corpus_dir: Annotated[
        str,
        typer.Option(
            "--corpus-dir",
            "-c",
            help="Directory containing topic JSON files",
        ),
    ] = DEFAULT_CORPUS_DIR,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Limit number of topics to benchmark (e.g. 3)"),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Optional single query to benchmark directly"),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output file path for markdown report"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Benchmark Single-Agent vs Multi-Agent across the offline corpus or a query."""
    _init()
    metrics: list[BenchmarkMetrics] = []

    if query:
        console.print(f"[bold cyan]Running benchmark on query:[/bold cyan] {query}")
        _, base_m = run_benchmark("Single-Agent Baseline", query, run_single_agent_baseline)
        _, multi_m = run_benchmark("Multi-Agent Workflow", query, run_multi_agent_workflow)
        metrics.extend([base_m, multi_m])
    else:
        console.print(
            f"[bold cyan]Running corpus benchmark from:[/bold cyan] {corpus_dir} (limit={limit})"
        )
        metrics = run_corpus_benchmark(corpus_dir=corpus_dir, limit=limit)

    report_md = render_markdown_report(metrics)

    # Save to output
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    console.print(Markdown(report_md))
    console.print(f"[bold green]Report saved successfully to:[/bold green] {output}")


if __name__ == "__main__":
    app()
