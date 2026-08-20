"""Unit tests for SupervisorAgent routing policy."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_first() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "researcher"
    assert updated.iteration == 1


def test_supervisor_routes_to_analyst_after_research() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Paper 1", snippet="Snippet 1")],
        research_notes="Found notes on multi-agent architectures.",
    )
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_after_analysis() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Paper 1", snippet="Snippet 1")],
        research_notes="Found notes on multi-agent architectures.",
        analysis_notes="Analyzed trade-offs between centralized and decentralized agents.",
    )
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "writer"


def test_supervisor_routes_to_done_when_answer_ready() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Paper 1", snippet="Snippet 1")],
        research_notes="Found notes.",
        analysis_notes="Analyzed notes.",
        final_answer="Comprehensive report on multi-agent systems.",
    )
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "done"


def test_supervisor_enforces_max_iterations() -> None:
    supervisor = SupervisorAgent(max_iterations=3)
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=3,
    )
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "done"
