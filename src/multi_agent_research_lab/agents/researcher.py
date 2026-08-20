"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("ResearcherAgent gathering sources for query: %s", state.request.query)

        # 1. Fetch sources
        sources = self.search_client.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources.extend(sources)

        # 2. Synthesize research notes via LLM
        sources_text = "\n\n".join(
            f"[{i + 1}] Title: {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
            for i, s in enumerate(state.sources)
        )

        system_prompt = (
            "You are an expert academic and industry AI researcher. "
            "Read raw source snippets and produce structured, factual research notes "
            "highlighting key concepts, state-of-the-art developments, and evidence."
        )
        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Sources gathered:\n{sources_text}\n\n"
            "Produce comprehensive bulleted research notes citing sources [1], [2], etc."
        )

        response = self.llm_client.complete(system_prompt, user_prompt, temperature=0.2)
        state.research_notes = response.content

        # 3. Record agent output and trace
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(state.sources),
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            name="researcher_finished",
            payload={"sources_count": len(state.sources), "cost_usd": response.cost_usd},
        )
        return state
