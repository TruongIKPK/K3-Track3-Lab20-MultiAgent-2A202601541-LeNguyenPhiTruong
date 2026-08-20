"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` with deep comparative analysis."""
        logger.info("AnalystAgent analyzing research findings for: %s", state.request.query)

        system_prompt = (
            "You are a Senior Systems and AI Research Analyst. "
            "Your task is to critically analyze raw research notes, extract key trade-offs, "
            "evaluate evidence strength, compare methodologies, and highlight failure modes."
        )
        user_prompt = (
            f"Original Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{state.research_notes or 'No raw notes provided.'}\n\n"
            "Please provide a structured analysis containing:\n"
            "1. Core Technical Foundations & Mechanisms\n"
            "2. Trade-offs & Comparisons (Latency, Cost, Accuracy, Scalability)\n"
            "3. Failure Modes & Production Guardrails\n"
            "4. Synthesis of Strengths & Limitations"
        )

        response = self.llm_client.complete(system_prompt, user_prompt, temperature=0.1)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event(
            name="analyst_finished",
            payload={"cost_usd": response.cost_usd},
        )
        return state
