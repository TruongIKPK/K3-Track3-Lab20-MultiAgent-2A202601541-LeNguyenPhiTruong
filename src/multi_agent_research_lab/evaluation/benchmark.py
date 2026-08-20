"""Benchmark suite for comparing single-agent baseline vs multi-agent research workflow."""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_single_agent_baseline(query: str, audience: str = "technical learners") -> ResearchState:
    """Execute a single-agent baseline completion."""
    client = LLMClient()
    state = ResearchState(request=ResearchQuery(query=query, audience=audience))

    system_prompt = (
        "You are an expert AI research assistant. Provide a comprehensive technical report "
        "answering the user's research query directly, covering architecture and trade-offs."
    )
    user_prompt = (
        f"Query: {query}\nTarget Audience: {audience}\n\nWrite a detailed research report."
    )

    response = client.complete(system_prompt, user_prompt, temperature=0.3)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={"cost_usd": response.cost_usd},
        )
    )
    state.record_route("single_agent")
    return state


def run_multi_agent_workflow(query: str, audience: str = "technical learners") -> ResearchState:
    """Execute the full multi-agent workflow."""
    state = ResearchState(request=ResearchQuery(query=query, audience=audience))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def calculate_citation_coverage(answer: str | None, expected_sources_count: int = 3) -> float:
    """Calculate citation coverage based on bracketed references like [1], [A01], [source_id]."""
    if not answer:
        return 0.0
    citations = re.findall(r"\[([A-Za-z0-9_\-]+)\]", answer)
    unique_citations = set(citations)
    if not unique_citations:
        return 0.0
    return min(1.0, len(unique_citations) / max(1, expected_sources_count))


def calculate_quality_score(
    state: ResearchState,
    citation_coverage: float,
    is_multi_agent: bool,
) -> float:
    """Rigorous quality scoring aligned with benchmark rubric (scale 0-10).

    Components:
    - Base completion: 2.0 pts
    - Structure & Clarity (# Headings, Executive Summary): 1.5 pts
    - Technical Depth & Word Count: 1.5 pts
    - Mechanisms & Trade-offs: 1.5 pts
    - Failure Modes & Guardrails: 1.0 pts
    - Grounded Citations (citation_coverage * 2.0): up to 2.0 pts
    - Multi-Agent Synthesis & Verification Handoff: 0.5 pts
    """
    if not state.final_answer:
        return 0.0

    score = 2.0  # Base score
    text = state.final_answer.lower()

    # 1. Structure & Clarity (up to 1.5 pts)
    if "#" in state.final_answer and "summary" in text:
        score += 1.5
    elif "#" in state.final_answer:
        score += 1.0

    # 2. Depth & Length (up to 1.5 pts)
    word_count = len(state.final_answer.split())
    if word_count >= 600:
        score += 1.5
    elif word_count >= 300:
        score += 1.0
    elif word_count >= 150:
        score += 0.5

    # 3. Mechanisms & Trade-offs (up to 1.5 pts)
    if "trade-off" in text or "tradeoff" in text or "comparison" in text:
        score += 1.5
    elif "architecture" in text or "latency" in text or "cost" in text:
        score += 0.8

    # 4. Failure Modes & Production Guardrails (up to 1.0 pt)
    if ("guardrail" in text or "safety" in text) and ("failure" in text or "limitation" in text):
        score += 1.0
    elif "failure" in text or "guardrail" in text or "limitation" in text:
        score += 0.5

    # 5. Citation Grounding & Fact Verification (up to 2.0 pts)
    score += min(2.0, citation_coverage * 2.0)

    # 6. Multi-Agent Research Handoff & Independence (0.5 pt)
    if is_multi_agent and state.research_notes and state.analysis_notes:
        score += 0.5

    return min(10.0, round(score, 1))


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    expected_sources: int = 3,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a runner function, measure latency, token cost, citation coverage, and quality."""
    started = perf_counter()
    try:
        state = runner(query)
        failure_rate = 0.0 if state.final_answer else 1.0
    except Exception as exc:
        logger.error("Benchmark run failed for %s: %s", run_name, exc)
        state = ResearchState(request=ResearchQuery(query=query), errors=[str(exc)])
        failure_rate = 1.0

    latency = perf_counter() - started

    # Aggregate cost
    total_cost = sum(result.metadata.get("cost_usd", 0.0) or 0.0 for result in state.agent_results)
    is_multi = "multi" in run_name.lower()
    citation_cov = calculate_citation_coverage(state.final_answer, expected_sources)
    quality = calculate_quality_score(
        state, citation_coverage=citation_cov, is_multi_agent=is_multi
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0 else None,
        quality_score=quality if failure_rate == 0.0 else 0.0,
        citation_coverage=citation_cov,
        failure_rate=failure_rate,
        notes=f"Iterations: {state.iteration}, Sources: {len(state.sources)}",
    )
    return state, metrics


def run_corpus_benchmark(
    corpus_dir: Path | str,
    limit: int | None = None,
) -> list[BenchmarkMetrics]:
    """Run benchmark over offline corpus topic files."""
    path = Path(corpus_dir)
    topic_files = sorted(path.glob("*.json"))
    if limit:
        topic_files = topic_files[:limit]

    all_metrics: list[BenchmarkMetrics] = []

    for file_path in topic_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except Exception as exc:
            logger.warning("Could not read topic file %s: %s", file_path, exc)
            continue

        topic_name = data.get("topic", {}).get("name", file_path.stem)
        query = data.get("topic", {}).get("research_question", topic_name)
        sources_count = len(data.get("knowledge_base", {}).get("source_documents", [])) or 3

        logger.info("Evaluating topic: %s", topic_name)

        # 1. Run Baseline
        _, base_metric = run_benchmark(
            run_name=f"Baseline: {topic_name[:30]}...",
            query=query,
            runner=run_single_agent_baseline,
            expected_sources=sources_count,
        )
        all_metrics.append(base_metric)

        # 2. Run Multi-Agent
        _, multi_metric = run_benchmark(
            run_name=f"Multi-Agent: {topic_name[:30]}...",
            query=query,
            runner=run_multi_agent_workflow,
            expected_sources=sources_count,
        )
        all_metrics.append(multi_metric)

    return all_metrics
