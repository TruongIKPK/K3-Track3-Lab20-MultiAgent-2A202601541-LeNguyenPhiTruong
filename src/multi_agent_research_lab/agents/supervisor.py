import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        self.max_iterations = max_iterations or get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect shared state and determine the next agent route.

        Routing policy:
        1. Check max iterations limit to prevent infinite loops -> "done"
        2. If research_notes or sources are missing -> "researcher"
        3. If analysis_notes are missing -> "analyst"
        4. If final_answer is missing -> "writer"
        5. If final_answer is present -> "done"
        """
        next_route = self.decide_next_route(state)

        state.record_route(next_route)
        state.add_trace_event(
            name="supervisor_decision",
            payload={
                "iteration": state.iteration,
                "next_agent": next_route,
                "has_research": bool(state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        logger.info(
            "Supervisor routing decision (iteration %d): -> %s",
            state.iteration,
            next_route,
        )
        return state
    # Chỉ là Supervisor quyết định bằng code chứ chưa dùng LLM để quyết định 
    # Nếu dùng LLM để quyết định thì sẽ dùng thêm 1 agent nữa là router agent và supervisor sẽ là người điều phối toàn bộ các agent
    def decide_next_route(self, state: ResearchState) -> str:
        """Determine the next step based on state fields and guardrails."""
        # 1. Guardrail: Max iterations exceeded
        if state.iteration >= self.max_iterations:
            logger.warning(
                "Max iterations (%d) reached. Terminating workflow.",
                self.max_iterations,
            )
            return "done"

        # 2. Researcher step: need sources / research notes
        if not state.research_notes and not state.sources:
            return AgentName.RESEARCHER.value

        # 3. Analyst step: need analysis on research notes
        if not state.analysis_notes:
            return AgentName.ANALYST.value

        # 4. Writer step: need final synthesis answer
        if not state.final_answer:
            return AgentName.WRITER.value

        # 5. Finished
        return "done"

