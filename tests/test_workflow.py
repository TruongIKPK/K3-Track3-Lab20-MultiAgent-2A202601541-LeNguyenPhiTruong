"""Tests for MultiAgentWorkflow graph construction and execution."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class DummyResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        state.sources.append(SourceDocument(title="Doc 1", snippet="Snippet 1"))
        state.research_notes = "Dummy research notes."
        return state


class DummyAnalyst(BaseAgent):
    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        state.analysis_notes = "Dummy structured analysis."
        return state


class DummyWriter(BaseAgent):
    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        state.final_answer = "Dummy comprehensive final report."
        return state


def test_workflow_builds_and_runs_end_to_end() -> None:
    workflow = MultiAgentWorkflow(
        supervisor=SupervisorAgent(),
        researcher=DummyResearcher(),  # type: ignore[arg-type]
        analyst=DummyAnalyst(),  # type: ignore[arg-type]
        writer=DummyWriter(),  # type: ignore[arg-type]
    )

    initial_state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent orchestration patterns")
    )

    result_state = workflow.run(initial_state)

    assert result_state.final_answer == "Dummy comprehensive final report."
    assert result_state.analysis_notes == "Dummy structured analysis."
    assert result_state.research_notes == "Dummy research notes."
    assert result_state.route_history == ["researcher", "analyst", "writer", "done"]
    assert result_state.iteration == 4
