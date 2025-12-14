import pytest

from recruiting_agent_pollock.memory import InterviewMemory, MemoryType
from recruiting_agent_pollock.orchestrator import CandidateProfile, InterviewConfig, InterviewOrchestrator


def test_memory_hypothesis_roundtrip() -> None:
    memory = InterviewMemory()

    entry = memory.add_or_update_hypothesis(
        statement="Candidate may have limited exposure to SQL (not mentioned yet).",
        confidence=0.25,
        basis=["Required skill has not been mentioned so far."],
        defeaters=["Not yet discussed; candidate may still have this experience."],
        related_skills=["SQL"],
        source="test",
    )

    assert entry.memory_type == MemoryType.HYPOTHESIS
    assert "statement" in entry.content

    llm_text = memory.get_hypotheses_for_llm(max_items=5)
    assert "unconfirmed" not in llm_text.lower()  # formatting is bullet-based, not narrative
    assert "Candidate may have limited exposure" in llm_text


@pytest.mark.asyncio
async def test_orchestrator_adds_hypotheses_from_missing_required_skills() -> None:
    orchestrator = InterviewOrchestrator()
    config = InterviewConfig(job_id="t1", job_title="Engineer", required_skills=["SQL", "Python"])
    candidate = CandidateProfile(name="A", email="a@example.com")

    await orchestrator.start_interview(config=config, candidate=candidate)

    # Candidate mentions neither required skill; engine should propose a low-confidence gap hypothesis.
    await orchestrator.process_candidate_input("I have worked on various backend services.")

    memory = orchestrator.memory
    assert memory is not None

    hyps = memory.get_entries_by_type(MemoryType.HYPOTHESIS)
    assert hyps, "Expected at least one hypothesis entry"
    assert any("not mentioned" in (h.content.get("statement", "").lower()) for h in hyps)


@pytest.mark.asyncio
async def test_hypothesis_auto_rejects_when_skill_later_mentioned() -> None:
    orchestrator = InterviewOrchestrator()
    config = InterviewConfig(job_id="t2", job_title="Engineer", required_skills=["SQL"])
    candidate = CandidateProfile(name="B", email="b@example.com")

    await orchestrator.start_interview(config=config, candidate=candidate)

    await orchestrator.process_candidate_input("I have worked on backend services.")
    memory = orchestrator.memory
    assert memory is not None

    # Now mention the skill; the earlier "not mentioned" hypothesis should be rejected.
    await orchestrator.process_candidate_input("I used SQL for reporting and ETL.")

    hyps = memory.get_entries_by_type(MemoryType.HYPOTHESIS)
    target = [h for h in hyps if "sql" in (h.content.get("statement", "").lower()) and "not mentioned" in (h.content.get("statement", "").lower())]
    assert target
    assert target[0].content.get("status") == "rejected"


@pytest.mark.asyncio
async def test_hypothesis_confirms_on_explicit_uncertainty() -> None:
    orchestrator = InterviewOrchestrator()
    config = InterviewConfig(job_id="t3", job_title="Engineer", required_skills=["SQL"])
    candidate = CandidateProfile(name="C", email="c@example.com")

    await orchestrator.start_interview(config=config, candidate=candidate)

    # Parser is LLM-based and might not extract skills in a deterministic way,
    # so mention the skill explicitly alongside the negation.
    await orchestrator.process_candidate_input("I don't know SQL and haven't used it before.")

    memory = orchestrator.memory
    assert memory is not None
    hyps = memory.get_entries_by_type(MemoryType.HYPOTHESIS)
    assert any(
        (h.content.get("status") == "confirmed")
        and ("self-reported uncertainty" in (h.content.get("statement", "").lower()))
        for h in hyps
    )
