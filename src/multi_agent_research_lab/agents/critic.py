"""Critic agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking, quality assessment, and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append audit findings."""
        logger.info("CriticAgent reviewing final answer for: %s", state.request.query)

        system_prompt = (
            "You are an AI Quality & Fact-Checking Auditor. "
            "Review the final answer against research notes and sources to verify:\n"
            "1. Factuality & Hallucination: Are assertions supported by sources?\n"
            "2. Citation Coverage: Are inline citations accurate and properly referenced?\n"
            "3. Completeness & Clarity: Does it thoroughly address the initial query?"
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Final Answer:\n{state.final_answer or 'N/A'}\n\n"
            "Provide a brief evaluation summary (Score 1-10, strengths, and flagged issues if any)."
        )

        response = self.llm_client.complete(system_prompt, user_prompt, temperature=0.0)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={"cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event(
            name="critic_finished",
            payload={"cost_usd": response.cost_usd},
        )
        return state
