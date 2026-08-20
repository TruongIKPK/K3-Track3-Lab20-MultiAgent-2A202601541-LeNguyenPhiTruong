"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with publication-ready report."""
        logger.info("WriterAgent generating final answer for: %s", state.request.query)

        sources_ref = "\n".join(
            f"[{i + 1}] {s.title} - {s.url or 'Internal Source'}"
            for i, s in enumerate(state.sources)
        )

        system_prompt = (
            "You are a Principal AI Technical Writer. "
            "Write a coherent, publication-grade technical report in Markdown. "
            "Integrate both empirical research notes and structured analysis into a guide. "
            "Include proper inline citations [1], [2] referencing the bibliography at the end."
        )
        user_prompt = (
            f"Query / Topic: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"=== RESEARCH NOTES ===\n{state.research_notes or 'N/A'}\n\n"
            f"=== ANALYSIS NOTES ===\n{state.analysis_notes or 'N/A'}\n\n"
            f"=== SOURCES REFERENCE ===\n{sources_ref or 'N/A'}\n\n"
            "Format requirements:\n"
            "- # Title and Executive Summary\n"
            "- ## Key Architectures & Technical Foundations\n"
            "- ## Comparative Analysis & Trade-offs\n"
            "- ## Production Guardrails & Best Practices\n"
            "- ## References & Citations"
        )

        response = self.llm_client.complete(system_prompt, user_prompt, temperature=0.4)
        state.final_answer = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event(
            name="writer_finished",
            payload={"cost_usd": response.cost_usd},
        )
        return state
