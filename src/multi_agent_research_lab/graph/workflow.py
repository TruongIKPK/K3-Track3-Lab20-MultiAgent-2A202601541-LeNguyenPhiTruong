"""LangGraph workflow for multi-agent research."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Orchestrates:
    - Supervisor: inspects state, routes to worker or ends workflow.
    - Researcher: searches and gathers source documents & research notes.
    - Analyst: synthesizes research into structured analysis notes.
    - Writer: writes the final comprehensive answer with citations.
    - (Optional) Critic: reviews and validates output.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.supervisor = supervisor or SupervisorAgent()
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled_graph: CompiledStateGraph | None = None

    def _to_state(self, raw_state: ResearchState | dict[str, Any]) -> ResearchState:
        if isinstance(raw_state, ResearchState):
            return raw_state
        return ResearchState.model_validate(raw_state)

    def _route_condition(self, state: ResearchState | dict[str, Any]) -> str:
        """Route to the next node based on supervisor's latest routing decision."""
        history = state.get("route_history", []) if isinstance(state, dict) else state.route_history

        if not history:
            return END

        next_target = history[-1]
        if next_target == "done":
            return END
        return next_target

    def build(self) -> CompiledStateGraph:
        """Create and compile the LangGraph multi-agent graph."""
        builder = StateGraph(ResearchState)

        # 1. Add nodes
        builder.add_node("supervisor", lambda s: self.supervisor.run(self._to_state(s)))
        builder.add_node("researcher", lambda s: self.researcher.run(self._to_state(s)))
        builder.add_node("analyst", lambda s: self.analyst.run(self._to_state(s)))
        builder.add_node("writer", lambda s: self.writer.run(self._to_state(s)))
        builder.add_node("critic", lambda s: self.critic.run(self._to_state(s)))

        # 2. Entrypoint -> supervisor
        builder.add_edge(START, "supervisor")

        # 3. Conditional routing from supervisor
        builder.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                AgentName.RESEARCHER.value: "researcher",
                AgentName.ANALYST.value: "analyst",
                AgentName.WRITER.value: "writer",
                AgentName.CRITIC.value: "critic",
                "done": END,
                END: END,
            },
        )

        # 4. Workers hand back control to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        builder.add_edge("critic", "supervisor")

        self._compiled_graph = builder.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the multi-agent graph and return final state."""
        if self._compiled_graph is None:
            self.build()

        assert self._compiled_graph is not None
        output = self._compiled_graph.invoke(state)
        return self._to_state(output)

