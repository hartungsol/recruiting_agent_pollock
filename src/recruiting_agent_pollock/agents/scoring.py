"""
Scoring agent.

Evaluates candidate responses and maintains running assessments
of skills and overall fit.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState
from recruiting_agent_pollock.orchestrator.schemas import SkillAssessment

if TYPE_CHECKING:
    from recruiting_agent_pollock.memory import InterviewMemory

logger = logging.getLogger(__name__)


class ScoreUpdate(BaseModel):
    """Update to a candidate's scores after processing a response."""

    skill_updates: list[SkillAssessment] = Field(
        default_factory=list,
        description="Updated skill assessments",
    )
    overall_impression_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Change to overall impression (-1 to +1)",
    )
    reasoning: str = Field(
        default="",
        description="Explanation for score updates",
    )


class FinalScore(BaseModel):
    """Final scoring of a candidate."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall candidate score (0-1)",
    )
    skill_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual skill scores",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Identified strengths",
    )
    areas_for_development: list[str] = Field(
        default_factory=list,
        description="Areas needing development",
    )
    fit_assessment: str = Field(
        default="",
        description="Assessment of role fit",
    )
    recommendation: str = Field(
        default="",
        description="Hiring recommendation",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment",
    )


class ScoringAgentBase(ABC):
    """Abstract base class for scoring agents."""

    @abstractmethod
    async def update_scores(
        self,
        state: InterviewState,
        input_text: str,
    ) -> ScoreUpdate:
        """
        Update scores based on a candidate response.

        Args:
            state: Current interview state.
            input_text: Candidate's response.

        Returns:
            Score updates to apply.
        """
        ...

    @abstractmethod
    async def calculate_final_score(
        self,
        state: InterviewState,
    ) -> FinalScore:
        """
        Calculate final scores at the end of the interview.

        Args:
            state: Complete interview state.

        Returns:
            Final scoring and recommendation.
        """
        ...


class ScoringAgent(ScoringAgentBase):
    """
    LLM-based scoring agent with defeasible reasoning.

    Evaluates candidate responses against job requirements and uses
    defeasible reasoning to weight evidence and handle conflicting signals.
    """

    UPDATE_PROMPT = """You are a scoring agent for a recruiting interview system.
Evaluate the candidate's latest response and provide score updates.

Job context:
- Position: {job_title}
- Required skills: {required_skills}
- Interview phase: {phase}

Current skill assessments:
{current_assessments}

Unconfirmed hypotheses (soft signals ONLY; do not treat as facts):
{hypotheses_context}

Latest candidate response:
"{input_text}"

Evaluate this response and return a JSON object:
{{
    "skill_updates": [
        {{
            "skill_name": "<skill>",
            "score": <0.0-1.0>,
            "evidence": "<supporting evidence from response>",
            "confidence": <0.0-1.0>
        }}
    ],
    "overall_impression_delta": <-1.0 to +1.0 change>,
    "reasoning": "<brief explanation of scoring>"
}}

Be fair and objective. Score based on demonstrated competence, not style.
Only return valid JSON, no other text."""

    FINAL_SCORE_PROMPT = """You are a scoring agent providing final assessment for a recruiting interview.

Job context:
- Position: {job_title}
- Required skills: {required_skills}

Complete interview summary:
- Total exchanges: {total_exchanges}
- Phases completed: {phases_completed}

Skill assessments collected:
{skill_assessments}

Red flags detected:
{red_flags}

Conversation highlights:
{conversation_highlights}

Provide a comprehensive final assessment. Return a JSON object:
{{
    "overall_score": <0.0-1.0>,
    "skill_scores": {{
        "<skill1>": <0.0-1.0>,
        "<skill2>": <0.0-1.0>
    }},
    "strengths": ["<strength1>", "<strength2>", ...],
    "areas_for_development": ["<area1>", "<area2>", ...],
    "fit_assessment": "<paragraph assessing role fit>",
    "recommendation": "<strong_yes|yes|maybe|no|strong_no>",
    "confidence": <0.0-1.0 in this assessment>
}}

Be fair, balanced, and evidence-based.
Only return valid JSON, no other text."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        # oscar_adapter: OSCARAdapter | None = None,  # Future: for weighted evaluation
    ) -> None:
        """
        Initialize the scoring agent.

        Args:
            llm_client: LLM client for evaluation. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()
        # self._oscar_adapter = oscar_adapter  # Future

    def _format_current_assessments(self, state: InterviewState) -> str:
        """Format current skill assessments for prompt."""
        if not state.skill_assessments:
            return "(No assessments yet)"
        
        lines = []
        for assessment in state.skill_assessments:
            lines.append(f"  - {assessment.skill_name}: {assessment.score:.2f} (confidence: {assessment.confidence:.2f})")
        return "\n".join(lines)

    def _format_assessments_from_memory(self, memory: InterviewMemory) -> str:
        """Format skill assessments from memory."""
        from recruiting_agent_pollock.memory import MemoryType
        
        assessments = memory.get_entries_by_type(MemoryType.SKILL_ASSESSMENT)
        if not assessments:
            return "(No assessments yet)"
        
        # Get latest assessment per skill
        latest_by_skill: dict[str, dict] = {}
        for entry in assessments:
            skill = entry.content.get("skill_name", "").lower()
            if skill:
                latest_by_skill[skill] = entry.content
        
        lines = []
        for skill, content in latest_by_skill.items():
            score = content.get("score", 0.5)
            confidence = content.get("confidence", 0.5)
            lines.append(f"  - {skill}: {score:.2f} (confidence: {confidence:.2f})")
        return "\n".join(lines) if lines else "(No assessments yet)"

    def _get_conversation_highlights(self, state: InterviewState) -> str:
        """Extract key highlights from conversation."""
        candidate_responses = [
            e.get("content", "")[:300]
            for e in state.conversation_history
            if e.get("role") == "candidate"
        ]
        
        if not candidate_responses:
            return "(No candidate responses)"
        
        # Take first, middle, and last responses as highlights
        highlights = []
        if len(candidate_responses) >= 1:
            highlights.append(f"Opening: {candidate_responses[0]}")
        if len(candidate_responses) >= 3:
            mid = len(candidate_responses) // 2
            highlights.append(f"Middle: {candidate_responses[mid]}")
        if len(candidate_responses) >= 2:
            highlights.append(f"Recent: {candidate_responses[-1]}")
        
        return "\n".join(highlights)

    async def update_scores(
        self,
        state: InterviewState,
        input_text: str,
        memory: InterviewMemory | None = None,
    ) -> ScoreUpdate:
        """
        Update scores based on a candidate response.

        Args:
            state: Current interview state.
            input_text: Candidate's response.
            memory: Optional interview memory for enhanced context.

        Returns:
            Score updates to apply.
        """
        required_skills = ", ".join(state.config.required_skills[:8]) if state.config.required_skills else "Not specified"
        
        # Use memory for current assessments if available
        if memory:
            current_assessments = self._format_assessments_from_memory(memory)
            sanitized_input = memory._sanitize_for_llm(input_text, max_length=600)
            hypotheses_context = memory.get_hypotheses_for_llm(max_items=3)
        else:
            current_assessments = self._format_current_assessments(state)
            sanitized_input = input_text[:600] if len(input_text) > 600 else input_text
            hypotheses_context = "(None)"
        
        prompt = self.UPDATE_PROMPT.format(
            job_title=state.config.job_title,
            required_skills=required_skills,
            phase=state.current_phase.value,
            current_assessments=current_assessments,
            hypotheses_context=hypotheses_context,
            input_text=sanitized_input,
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            raw_skill_updates = response.get("skill_updates") or []
            if not isinstance(raw_skill_updates, list):
                raw_skill_updates = []

            skill_updates: list[SkillAssessment] = []
            for update in raw_skill_updates:
                if not isinstance(update, dict):
                    continue
                skill_updates.append(
                    SkillAssessment(
                        skill_name=update.get("skill_name", "unknown"),
                        score=max(0.0, min(1.0, float(update.get("score", 0.5)))),
                        evidence=update.get("evidence", ""),
                        confidence=max(0.0, min(1.0, float(update.get("confidence", 0.5)))),
                    )
                )

            try:
                delta = float(response.get("overall_impression_delta", 0.0))
            except (TypeError, ValueError):
                delta = 0.0
            delta = max(-1.0, min(1.0, delta))

            return ScoreUpdate(
                skill_updates=skill_updates,
                overall_impression_delta=delta,
                reasoning=response.get("reasoning", ""),
            )

        except OllamaError as e:
            logger.error(f"LLM scoring failed: {e}")
            return ScoreUpdate(
                skill_updates=[],
                overall_impression_delta=0.0,
                reasoning=f"Scoring unavailable: {e}",
            )

    async def calculate_final_score(
        self,
        state: InterviewState,
    ) -> FinalScore:
        """
        Calculate final scores at the end of the interview.

        Args:
            state: Complete interview state.

        Returns:
            Final scoring and recommendation.
        """
        required_skills = ", ".join(state.config.required_skills[:8]) if state.config.required_skills else "Not specified"
        
        # Format skill assessments
        skill_assessment_str = self._format_current_assessments(state)
        
        # Format red flags
        if state.red_flags:
            red_flag_str = "\n".join([
                f"  - {rf.category}: {rf.description} (severity: {rf.severity:.2f})"
                for rf in state.red_flags
            ])
        else:
            red_flag_str = "(None detected)"
        
        # Get phases completed
        phases_completed = [state.current_phase.value]  # Simplified - could track all phases
        
        prompt = self.FINAL_SCORE_PROMPT.format(
            job_title=state.config.job_title,
            required_skills=required_skills,
            total_exchanges=len(state.conversation_history),
            phases_completed=", ".join(phases_completed),
            skill_assessments=skill_assessment_str,
            red_flags=red_flag_str,
            conversation_highlights=self._get_conversation_highlights(state),
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            # Parse skill scores
            skill_scores = {}
            for skill, score in response.get("skill_scores", {}).items():
                skill_scores[skill] = max(0.0, min(1.0, float(score)))

            # Map recommendation to descriptive text
            rec_raw = response.get("recommendation") or "maybe"
            recommendation_map = {
                "strong_yes": "Strongly Recommend - Excellent candidate",
                "yes": "Recommend - Good candidate",
                "maybe": "Consider - Mixed signals, needs further evaluation",
                "no": "Do Not Recommend - Significant concerns",
                "strong_no": "Strongly Do Not Recommend - Major disqualifiers",
            }
            rec_key = str(rec_raw)
            recommendation = recommendation_map.get(rec_key, rec_key)

            return FinalScore(
                overall_score=max(0.0, min(1.0, float(response.get("overall_score", 0.5)))),
                skill_scores=skill_scores,
                strengths=response.get("strengths", []),
                areas_for_development=response.get("areas_for_development", []),
                fit_assessment=response.get("fit_assessment", "Assessment pending"),
                recommendation=recommendation,
                confidence=max(0.0, min(1.0, float(response.get("confidence", 0.5)))),
            )

        except OllamaError as e:
            logger.error(f"Final scoring failed: {e}")
            return FinalScore(
                overall_score=0.5,
                skill_scores={},
                strengths=[],
                areas_for_development=[],
                fit_assessment=f"Assessment unavailable: {e}",
                recommendation="Further review needed - scoring error",
                confidence=0.0,
            )

    async def process(self, state: InterviewState, input_text: str) -> str:
        """
        Process method for Agent protocol compatibility.

        Args:
            state: Current interview state.
            input_text: Response to score.

        Returns:
            Summary of score updates.
        """
        update = await self.update_scores(state, input_text)
        return f"Score delta: {update.overall_impression_delta:+.2f}. {update.reasoning}"
