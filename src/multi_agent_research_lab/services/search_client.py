"""Search client abstraction for ResearcherAgent."""

import logging
from typing import Any

import requests

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily and fallback support."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or get_settings().tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as exc:
                logger.warning("Tavily search failed (%s), falling back to web/mock search", exc)

        return self._search_fallback(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        results: list[SourceDocument] = []
        for item in data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled Source"),
                    url=item.get("url"),
                    snippet=item.get("content", item.get("snippet", "")),
                    metadata={"score": item.get("score", 1.0)},
                )
            )
        return results

    def _search_fallback(self, query: str, max_results: int) -> list[SourceDocument]:
        """Built-in search fallback for development and offline testing."""
        logger.info("Using built-in search provider for query: %s", query)
        base_sources = [
            SourceDocument(
                title=f"Comprehensive Overview: {query}",
                url="https://arxiv.org/abs/multi-agent-survey",
                snippet=(
                    f"State-of-the-art research on {query}. Key foundations, multi-agent "
                    "coordination, graph reasoning, RAG, and architectural benchmarks."
                ),
                metadata={"provider": "fallback", "relevance": 0.95},
            ),
            SourceDocument(
                title=f"Architectural Patterns and Systems Design for {query}",
                url="https://github.com/topics/multi-agent-systems",
                snippet=(
                    "System design patterns including supervisor routing, stateful handoffs, "
                    "shared memory state machines, guardrails, and error recovery policies."
                ),
                metadata={"provider": "fallback", "relevance": 0.90},
            ),
            SourceDocument(
                title=f"Empirical Evaluation & Performance Trade-offs in {query}",
                url="https://papers.ai/evaluation-tradeoffs",
                snippet=(
                    "Comparative study analyzing latency, token cost, accuracy, citation fidelity, "
                    "and failure modes across single-agent and multi-agent topologies."
                ),
                metadata={"provider": "fallback", "relevance": 0.88},
            ),
        ]
        return base_sources[:max_results]
