"""
Interview state management.

Tracks the current state of an interview session, including conversation
history, assessments, and reasoning context.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from recruiting_agent_pollock.orchestrator.schemas import (
    CandidateProfile,
    InterviewConfig,
    InterviewPhase,
    InterviewResult,
    InterviewTurn,
    RedFlag,
    SkillAssessment,
    TurnRole,
)


class InterviewState:
    """
    Manages the mutable state of an interview session.

    This class tracks conversation history, current phase, assessments,
    and provides methods for state transitions.
    """

    def __init__(
        self,
        config: InterviewConfig,
        candidate: CandidateProfile,
    ) -> None:
        """
        Initialize interview state.

        Args:
            config: Interview configuration.
            candidate: Candidate profile information.
        """
        self._interview_id: UUID = uuid4()
        self._config = config
        self._candidate = candidate
        self._turns: list[InterviewTurn] = []
        self._current_phase: InterviewPhase = config.phases[0] if config.phases else InterviewPhase.INTRODUCTION
        self._phase_index: int = 0
        self._skill_assessments: list[SkillAssessment] = []
        self._red_flags: list[RedFlag] = []
        self._reasoning_trace: list[dict[str, Any]] = []
        self._started_at: datetime = datetime.now(timezone.utc)
        self._is_complete: bool = False
        
        # Interview pacing state
        self._core_questions_asked: int = 0
        self._followups_per_topic: dict[str, int] = {}  # topic -> count
        self._current_topic: str | None = None
        self._offered_continuation: bool = False
        self._candidate_wants_continue: bool = False
        
        # Experience-aware interviewing state
        self._experience_level: str = candidate.experience_level  # Initialize from profile
        self._experience_inferred: bool = candidate.experience_level != "unknown"
        self._closing_message_sent: bool = False  # Prevent duplicate closings

    @property
    def interview_id(self) -> UUID:
        """Get the unique interview identifier."""
        return self._interview_id

    @property
    def config(self) -> InterviewConfig:
        """Get the interview configuration."""
        return self._config

    @property
    def candidate(self) -> CandidateProfile:
        """Get the candidate profile."""
        return self._candidate

    @property
    def current_phase(self) -> InterviewPhase:
        """Get the current interview phase."""
        return self._current_phase

    @property
    def turns(self) -> list[InterviewTurn]:
        """Get all conversation turns."""
        return self._turns.copy()

    @property
    def conversation_history(self) -> list[dict[str, str]]:
        """Get conversation history in a format suitable for LLM context."""
        return [{"role": turn.role.value, "content": turn.content} for turn in self._turns]

    @property
    def skill_assessments(self) -> list[SkillAssessment]:
        """Get all skill assessments."""
        return self._skill_assessments.copy()

    @property
    def red_flags(self) -> list[RedFlag]:
        """Get all red flags."""
        return self._red_flags.copy()

    # Interview pacing properties
    @property
    def core_questions_asked(self) -> int:
        """Get the number of core questions asked."""
        return self._core_questions_asked

    @property
    def current_topic(self) -> str | None:
        """Get the current topic being probed."""
        return self._current_topic

    @property
    def offered_continuation(self) -> bool:
        """Check if continuation was offered to candidate."""
        return self._offered_continuation

    @property
    def candidate_wants_continue(self) -> bool:
        """Check if candidate wants to continue after core questions."""
        return self._candidate_wants_continue

    def should_offer_continuation(self) -> bool:
        """Check if we should offer the candidate option to continue."""
        if not self._config.offer_continuation:
            return False
        if self._offered_continuation:
            return False
        return self._core_questions_asked >= self._config.max_core_questions

    def mark_continuation_offered(self) -> None:
        """Mark that continuation was offered to candidate."""
        self._offered_continuation = True

    def set_candidate_wants_continue(self, wants_continue: bool) -> None:
        """Set whether candidate wants to continue."""
        self._candidate_wants_continue = wants_continue

    def increment_core_questions(self) -> None:
        """Increment the core questions counter."""
        self._core_questions_asked += 1

    def set_current_topic(self, topic: str | None) -> None:
        """Set the current topic being discussed."""
        self._current_topic = topic

    def get_followups_for_topic(self, topic: str) -> int:
        """Get the number of follow-ups asked for a specific topic."""
        return self._followups_per_topic.get(topic, 0)

    def increment_followup_for_topic(self, topic: str) -> int:
        """
        Increment follow-up count for a topic.
        
        Returns:
            The new follow-up count for this topic.
        """
        current = self._followups_per_topic.get(topic, 0)
        self._followups_per_topic[topic] = current + 1
        return current + 1

    def can_followup_on_topic(self, topic: str) -> bool:
        """Check if we can ask another follow-up on this topic."""
        return self.get_followups_for_topic(topic) < self._config.max_followups_per_topic

    # Experience-aware interviewing properties and methods
    @property
    def experience_level(self) -> str:
        """Get the candidate's experience level."""
        return self._experience_level

    @property
    def experience_inferred(self) -> bool:
        """Check if experience level has been determined."""
        return self._experience_inferred

    @property
    def closing_message_sent(self) -> bool:
        """Check if closing message has already been sent."""
        return self._closing_message_sent

    def mark_closing_sent(self) -> None:
        """Mark that the closing message has been sent."""
        self._closing_message_sent = True

    def set_experience_level(self, level: str) -> None:
        """
        Set the candidate's experience level.
        
        Args:
            level: Experience level (entry, mid, senior, unknown).
        """
        if level in ("entry", "mid", "senior", "unknown"):
            self._experience_level = level
            self._experience_inferred = True

    def is_entry_level(self) -> bool:
        """Check if candidate is entry-level."""
        return self._experience_level == "entry"

    @property
    def is_complete(self) -> bool:
        """Check if the interview is complete."""
        return self._is_complete

    def add_turn(
        self,
        role: TurnRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> InterviewTurn:
        """
        Add a new conversation turn.

        Args:
            role: Role of the speaker.
            content: Content of the turn.
            metadata: Optional additional metadata.

        Returns:
            The created InterviewTurn.
        """
        turn = InterviewTurn(
            role=role,
            content=content,
            phase=self._current_phase,
            metadata=metadata or {},
        )
        self._turns.append(turn)
        return turn

    def advance_phase(self) -> InterviewPhase | None:
        """
        Advance to the next interview phase.

        Returns:
            The new phase, or None if interview is complete.
        """
        if self._phase_index + 1 < len(self._config.phases):
            self._phase_index += 1
            self._current_phase = self._config.phases[self._phase_index]
            return self._current_phase
        return None

    def add_skill_assessment(self, assessment: SkillAssessment) -> None:
        """
        Add a skill assessment.

        Args:
            assessment: The skill assessment to add.
        """
        self._skill_assessments.append(assessment)

    def add_red_flag(self, red_flag: RedFlag) -> None:
        """
        Add a red flag.

        Args:
            red_flag: The red flag to add.
        """
        self._red_flags.append(red_flag)

    def add_reasoning_step(self, step: dict[str, Any]) -> None:
        """
        Add a reasoning trace step.

        Args:
            step: The reasoning step to record.
        """
        self._reasoning_trace.append(step)

    def complete(self, overall_score: float | None = None, recommendation: str = "") -> InterviewResult:
        """
        Mark the interview as complete and generate the result.

        Args:
            overall_score: Final overall score (0-1).
            recommendation: Final recommendation text.

        Returns:
            The complete InterviewResult.
        """
        self._is_complete = True
        return InterviewResult(
            interview_id=self._interview_id,
            candidate=self._candidate,
            config=self._config,
            turns=self._turns,
            skill_assessments=self._skill_assessments,
            red_flags=self._red_flags,
            overall_score=overall_score,
            recommendation=recommendation,
            reasoning_trace=self._reasoning_trace,
            started_at=self._started_at,
            completed_at=datetime.now(timezone.utc),
        )

    def get_conversation_context(self, max_turns: int | None = None) -> list[dict[str, str]]:
        """
        Get conversation history in a format suitable for LLM context.

        Args:
            max_turns: Maximum number of turns to include (None for all).

        Returns:
            List of role/content dicts for LLM consumption.
        """
        turns = self._turns[-max_turns:] if max_turns else self._turns
        return [{"role": turn.role.value, "content": turn.content} for turn in turns]
