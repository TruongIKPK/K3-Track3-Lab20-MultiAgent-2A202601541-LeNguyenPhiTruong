"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics and aggregated comparison to markdown."""
    lines = [
        "# Multi-Agent Systems Benchmark Report",
        "",
        "Evaluation comparing **Single-Agent Baseline** vs **Multi-Agent Research Workflow**.",
        "",
        "## Detailed Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    base_latencies: list[float] = []
    multi_latencies: list[float] = []
    base_costs: list[float] = []
    multi_costs: list[float] = []
    base_qualities: list[float] = []
    multi_qualities: list[float] = []
    base_citations: list[float] = []
    multi_citations: list[float] = []

    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"

        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f}s | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

        is_base = "baseline" in item.run_name.lower()
        target_latencies = base_latencies if is_base else multi_latencies
        target_costs = base_costs if is_base else multi_costs
        target_qualities = base_qualities if is_base else multi_qualities
        target_citations = base_citations if is_base else multi_citations

        target_latencies.append(item.latency_seconds)
        if item.estimated_cost_usd is not None:
            target_costs.append(item.estimated_cost_usd)
        if item.quality_score is not None:
            target_qualities.append(item.quality_score)
        if item.citation_coverage is not None:
            target_citations.append(item.citation_coverage)

    # Aggregated Summary
    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    lines.extend(
        [
            "",
            "## Summary Comparison",
            "",
            "| Metric | Single-Agent Baseline | Multi-Agent System | Delta / Trade-off |",
            "|---|---:|---:|---|",
            (
                f"| **Avg Latency** | {_avg(base_latencies):.2f}s | {_avg(multi_latencies):.2f}s "
                f"| Multi-agent overhead ({_avg(multi_latencies) - _avg(base_latencies):+.2f}s) |"
            ),
            (
                f"| **Avg Cost (USD)** | ${_avg(base_costs):.4f} | ${_avg(multi_costs):.4f} "
                f"| {_avg(multi_costs) / max(0.00001, _avg(base_costs)):.1f}x token utilization |"
            ),
            (
                f"| **Avg Quality Score** | {_avg(base_qualities):.1f}/10 "
                f"| {_avg(multi_qualities):.1f}/10 "
                f"| Quality gain ({_avg(multi_qualities) - _avg(base_qualities):+.1f} pts) |"
            ),
            (
                f"| **Avg Citation Coverage** | {_avg(base_citations):.0%} "
                f"| {_avg(multi_citations):.0%} "
                f"| Grounding improvement ({_avg(multi_citations) - _avg(base_citations):+.0%}) |"
            ),
            "",
        ]
    )

    return "\n".join(lines) + "\n"
