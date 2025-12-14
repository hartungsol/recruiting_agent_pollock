import pytest

from recruiting_agent_pollock.agents.parser_normalizer import ParserNormalizer
from recruiting_agent_pollock.orchestrator import CandidateProfile, InterviewConfig
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState


class FakeLLM:
    async def chat_with_json(self, messages, schema=None, temperature=0.2, **kwargs):
        return {
            "normalized_text": None,
            "key_points": [None, "A", "  ", 123],
            "mentioned_skills": [None, "Python"],
            "mentioned_experiences": [None, {"role": "Dev"}],
            "mentioned_achievements": [None, "Won award"],
            "time_references": [None, {"phrase": "3 years ago", "approximate_date": "2022"}],
            "sentiment": None,
            "confidence_indicators": [None, "maybe"],
        }


@pytest.mark.asyncio
async def test_parser_normalizer_drops_none_items() -> None:
    state = InterviewState(
        config=InterviewConfig(job_id="j1", job_title="Engineer", required_skills=["Python"]),
        candidate=CandidateProfile(name="X", email="x@example.com"),
    )

    parser = ParserNormalizer(llm_client=FakeLLM())
    parsed = await parser.parse(state, "hello")

    assert parsed.normalized_text == "hello"
    assert parsed.key_points == ["A"]
    assert parsed.mentioned_skills == ["Python"]
    assert parsed.confidence_indicators == ["maybe"]
    assert parsed.mentioned_experiences == [{"role": "Dev"}]
    assert parsed.time_references and isinstance(parsed.time_references[0], dict)
