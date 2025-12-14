"""
Smoke tests for the interview orchestrator.

Basic tests to verify the orchestrator can be instantiated and
run through a simple interview flow.
"""

import pytest

from recruiting_agent_pollock.orchestrator import (
    CandidateProfile,
    InterviewConfig,
    InterviewOrchestrator,
    InterviewState,
)
from recruiting_agent_pollock.orchestrator.schemas import InterviewPhase, TurnRole


class TestInterviewState:
    """Tests for InterviewState class."""

    @pytest.fixture
    def sample_config(self) -> InterviewConfig:
        """Create a sample interview configuration."""
        return InterviewConfig(
            job_id="test-job-001",
            job_title="Software Engineer",
            required_skills=["Python", "SQL"],
            preferred_skills=["FastAPI", "PostgreSQL"],
        )

    @pytest.fixture
    def sample_candidate(self) -> CandidateProfile:
        """Create a sample candidate profile."""
        return CandidateProfile(
            name="Jane Doe",
            email="jane.doe@example.com",
            resume_text="Experienced software engineer...",
        )

    @pytest.fixture
    def state(
        self,
        sample_config: InterviewConfig,
        sample_candidate: CandidateProfile,
    ) -> InterviewState:
        """Create an InterviewState instance."""
        return InterviewState(config=sample_config, candidate=sample_candidate)

    def test_state_initialization(
        self,
        state: InterviewState,
        sample_config: InterviewConfig,
        sample_candidate: CandidateProfile,
    ) -> None:
        """Test that state initializes correctly."""
        assert state.config == sample_config
        assert state.candidate == sample_candidate
        assert state.current_phase == InterviewPhase.INTRODUCTION
        assert len(state.turns) == 0
        assert not state.is_complete

    def test_add_turn(self, state: InterviewState) -> None:
        """Test adding conversation turns."""
        turn = state.add_turn(
            role=TurnRole.INTERVIEWER,
            content="Hello, welcome to the interview.",
        )

        assert turn.role == TurnRole.INTERVIEWER
        assert turn.content == "Hello, welcome to the interview."
        assert turn.phase == InterviewPhase.INTRODUCTION
        assert len(state.turns) == 1

    def test_advance_phase(self, state: InterviewState) -> None:
        """Test advancing interview phases."""
        initial_phase = state.current_phase
        new_phase = state.advance_phase()

        assert new_phase is not None
        assert new_phase != initial_phase
        assert state.current_phase == new_phase

    def test_complete_interview(self, state: InterviewState) -> None:
        """Test completing an interview."""
        state.add_turn(TurnRole.INTERVIEWER, "Hello!")
        state.add_turn(TurnRole.CANDIDATE, "Hi, nice to meet you.")

        result = state.complete(overall_score=0.75, recommendation="Proceed to next round")

        assert state.is_complete
        assert result.overall_score == 0.75
        assert result.recommendation == "Proceed to next round"
        assert len(result.turns) == 2

    def test_get_conversation_context(self, state: InterviewState) -> None:
        """Test getting conversation context for LLM."""
        state.add_turn(TurnRole.INTERVIEWER, "Question 1")
        state.add_turn(TurnRole.CANDIDATE, "Answer 1")
        state.add_turn(TurnRole.INTERVIEWER, "Question 2")

        context = state.get_conversation_context(max_turns=2)

        assert len(context) == 2
        assert context[0]["role"] == "candidate"
        assert context[1]["role"] == "interviewer"


class TestInterviewOrchestrator:
    """Tests for InterviewOrchestrator class."""

    @pytest.fixture
    def orchestrator(self) -> InterviewOrchestrator:
        """Create an InterviewOrchestrator instance."""
        return InterviewOrchestrator()

    @pytest.fixture
    def sample_config(self) -> InterviewConfig:
        """Create a sample interview configuration."""
        return InterviewConfig(
            job_id="test-job-001",
            job_title="Software Engineer",
        )

    @pytest.fixture
    def sample_candidate(self) -> CandidateProfile:
        """Create a sample candidate profile."""
        return CandidateProfile(
            name="John Smith",
            email="john.smith@example.com",
        )

    def test_orchestrator_initialization(
        self,
        orchestrator: InterviewOrchestrator,
    ) -> None:
        """Test orchestrator initializes correctly."""
        assert orchestrator.current_state is None
        assert not orchestrator.is_active

    @pytest.mark.asyncio
    async def test_start_interview(
        self,
        orchestrator: InterviewOrchestrator,
        sample_config: InterviewConfig,
        sample_candidate: CandidateProfile,
    ) -> None:
        """Test starting an interview."""
        introduction = await orchestrator.start_interview(
            config=sample_config,
            candidate=sample_candidate,
        )

        assert orchestrator.is_active
        assert orchestrator.current_state is not None
        assert sample_candidate.name in introduction
        assert sample_config.job_title in introduction

    @pytest.mark.asyncio
    async def test_process_candidate_input(
        self,
        orchestrator: InterviewOrchestrator,
        sample_config: InterviewConfig,
        sample_candidate: CandidateProfile,
    ) -> None:
        """Test processing candidate input."""
        await orchestrator.start_interview(sample_config, sample_candidate)

        response = await orchestrator.process_candidate_input(
            "I have 5 years of experience in software development."
        )

        assert response  # Should get some response
        assert orchestrator.current_state is not None
        assert len(orchestrator.current_state.turns) == 3  # intro + candidate + response

    @pytest.mark.asyncio
    async def test_process_without_active_interview(
        self,
        orchestrator: InterviewOrchestrator,
    ) -> None:
        """Test that processing fails without an active interview."""
        with pytest.raises(RuntimeError, match="No active interview"):
            await orchestrator.process_candidate_input("Hello")

    @pytest.mark.asyncio
    async def test_end_interview(
        self,
        orchestrator: InterviewOrchestrator,
        sample_config: InterviewConfig,
        sample_candidate: CandidateProfile,
    ) -> None:
        """Test ending an interview."""
        await orchestrator.start_interview(sample_config, sample_candidate)
        await orchestrator.process_candidate_input("I'm excited about this role.")

        result = await orchestrator.end_interview()

        assert not orchestrator.is_active
        assert orchestrator.current_state is None
        assert result.candidate.name == sample_candidate.name
        assert result.config.job_id == sample_config.job_id

    @pytest.mark.asyncio
    async def test_full_interview_flow(
        self,
        orchestrator: InterviewOrchestrator,
        sample_config: InterviewConfig,
        sample_candidate: CandidateProfile,
    ) -> None:
        """Test a complete interview flow."""
        # Start
        intro = await orchestrator.start_interview(sample_config, sample_candidate)
        assert intro

        # Multiple exchanges
        responses = []
        candidate_inputs = [
            "I have experience with Python and databases.",
            "My biggest achievement was leading a migration project.",
            "I handle pressure by prioritizing tasks effectively.",
        ]

        for input_text in candidate_inputs:
            response = await orchestrator.process_candidate_input(input_text)
            responses.append(response)
            assert response

        # End
        result = await orchestrator.end_interview()

        # Verify result
        expected_turns = 1 + (len(candidate_inputs) * 2)  # intro + (input + response) pairs
        assert len(result.turns) == expected_turns
        assert result.started_at is not None
        assert result.completed_at is not None


class TestInterviewSchemas:
    """Tests for Pydantic schemas."""

    def test_interview_config_defaults(self) -> None:
        """Test InterviewConfig with defaults."""
        config = InterviewConfig(
            job_id="test-001",
            job_title="Developer",
        )

        assert config.max_duration_minutes == 60
        assert config.required_skills == []
        assert len(config.phases) > 0

    def test_candidate_profile_auto_id(self) -> None:
        """Test CandidateProfile generates UUID."""
        profile = CandidateProfile(
            name="Test User",
            email="test@example.com",
        )

        assert profile.candidate_id is not None

    def test_interview_result_serialization(self) -> None:
        """Test InterviewResult can be serialized."""
        config = InterviewConfig(job_id="test", job_title="Test Role")
        candidate = CandidateProfile(name="Test", email="test@test.com")

        from recruiting_agent_pollock.orchestrator.schemas import InterviewResult

        result = InterviewResult(
            candidate=candidate,
            config=config,
        )

        # Should be JSON serializable
        json_data = result.model_dump_json()
        assert json_data
        assert "Test Role" in json_data
