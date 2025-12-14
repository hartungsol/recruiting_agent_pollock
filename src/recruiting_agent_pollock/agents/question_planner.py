"""
Question planner agent.

Plans and selects the next interview question based on the current
state, phase, and candidate responses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState
from recruiting_agent_pollock.orchestrator.schemas import InterviewPhase

if TYPE_CHECKING:
    from recruiting_agent_pollock.memory import InterviewMemory

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Types of interview questions."""

    OPEN_ENDED = "open_ended"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    SITUATIONAL = "situational"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    CLOSING = "closing"


class PlannedQuestion(BaseModel):
    """A planned interview question."""

    question_text: str = Field(..., description="The question to ask")
    question_type: QuestionType = Field(..., description="Type of question")
    target_skill: str | None = Field(default=None, description="Skill this question assesses")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority score")
    rationale: str = Field(default="", description="Why this question was selected")


class QuestionPlan(BaseModel):
    """A plan for upcoming questions."""

    next_question: PlannedQuestion = Field(..., description="Immediate next question")
    backup_questions: list[PlannedQuestion] = Field(
        default_factory=list,
        description="Alternative questions if needed",
    )
    should_transition_phase: bool = Field(
        default=False,
        description="Whether to transition to next phase",
    )
    recommended_phase: InterviewPhase | None = Field(
        default=None,
        description="Recommended next phase if transitioning",
    )
    should_wrap_up: bool = Field(
        default=False,
        description="Whether we have enough info and should wrap up",
    )


class QuestionPlannerBase(ABC):
    """Abstract base class for question planners."""

    @abstractmethod
    async def plan_next(
        self,
        state: InterviewState,
    ) -> QuestionPlan:
        """
        Plan the next question based on interview state.

        Args:
            state: Current interview state.

        Returns:
            Plan containing next question and alternatives.
        """
        ...


class QuestionPlanner(QuestionPlannerBase):
    """
    LLM-based question planner.

    Uses the LLM to dynamically plan interview questions based on
    the conversation history, candidate profile, and job requirements.
    """

    PLANNING_PROMPT = """You are a question planner for a FOCUSED, EFFICIENT recruiting interview.
Your goal is to assess the candidate with FEW, HIGH-IMPACT questions.

Job Information:
- Title: {job_title}
- Required skills: {required_skills}

Candidate Context:
- Experience level: {candidate_experience_level}
- This is a {experience_context}

Interview State:
- Current phase: {phase}
- Core questions asked: {core_questions_asked} of {max_core_questions} max
- Skills already assessed: {assessed_skills}
- Skills NOT yet assessed: {unassessed_skills}
- Current topic: {current_topic}
- Follow-ups on current topic: {followups_on_topic} of {max_followups} max
- Red flags detected: {red_flag_count}

Unconfirmed hypotheses (soft signals ONLY; do not treat as facts):
{hypotheses_context}

Recent conversation:
{conversation_context}

EXPERIENCE-AWARE RULES:
- For ENTRY-LEVEL: Ask about learning, coursework, projects, enthusiasm. Don't expect deep expertise.
- For MID-LEVEL: Ask about practical experience, problem-solving, growth.
- For SENIOR: Ask about leadership, architecture decisions, mentoring, strategy.
- Adapt question difficulty to candidate's level.

GENERAL RULES:
1. PRIORITIZE unassessed skills - don't repeat topics already covered
2. MAX {max_followups} follow-ups per topic, then MOVE ON to a new area
3. Ask DIRECT questions that assess multiple competencies when possible
4. If we've asked {max_core_questions} core questions, recommend wrapping up
5. DO NOT over-probe - get key info and move on

Question types:
- open_ended: General questions allowing free-form responses
- behavioral: "Tell me about a time when..." questions
- technical: Questions assessing technical knowledge
- follow_up: Probing deeper (use sparingly - max {max_followups} per topic)
- closing: Wrapping up the interview

Return JSON:
{{
    "next_question": {{
        "question_text": "<concise, focused question>",
        "question_type": "<type>",
        "target_skill": "<specific skill being assessed>",
        "priority": <0.0-1.0>,
        "rationale": "<why this is the most important question now>"
    }},
    "should_transition_phase": <true if this topic is sufficiently covered>,
    "recommended_phase": "<next phase or null>",
    "should_wrap_up": <true if we have enough info to assess candidate>
}}

Only return valid JSON."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the question planner.

        Args:
            llm_client: LLM client for question generation. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()

    def _build_conversation_context(self, state: InterviewState) -> str:
        """Build conversation context string from state."""
        history = state.conversation_history[-8:]
        if not history:
            return "(Interview just started)"
        
        lines = []
        for entry in history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:300]
            lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    def _get_assessed_skills(self, state: InterviewState) -> str:
        """Get list of skills that have been assessed."""
        if not state.skill_assessments:
            return "None yet"
        skills = [a.skill_name for a in state.skill_assessments]
        return ", ".join(skills[:10])

    def _get_unassessed_skills(self, state: InterviewState) -> str:
        """Get list of required skills not yet assessed."""
        assessed = {a.skill_name.lower() for a in state.skill_assessments}
        required = state.config.required_skills or []
        unassessed = [s for s in required if s.lower() not in assessed]
        if not unassessed:
            return "All required skills covered"
        return ", ".join(unassessed[:5])

    async def plan_next(
        self,
        state: InterviewState,
        memory: InterviewMemory | None = None,
    ) -> QuestionPlan:
        """
        Plan the next question based on interview state.

        Args:
            state: Current interview state.
            memory: Optional interview memory for enhanced context.

        Returns:
            Plan containing next question and alternatives.
        """
        required_skills = ", ".join(state.config.required_skills[:8]) if state.config.required_skills else "Not specified"
        
        # Get pacing info
        max_core = state.config.max_core_questions
        max_followups = state.config.max_followups_per_topic
        current_topic = state.current_topic or "None"
        followups_on_topic = state.get_followups_for_topic(current_topic) if state.current_topic else 0
        
        # Get experience context - prefer memory if available
        if memory:
            exp_level = memory.experience_level
        else:
            exp_level = state.experience_level
        experience_contexts = {
            "entry": "entry-level/junior candidate - focus on potential, learning ability, enthusiasm",
            "mid": "mid-level candidate - focus on practical experience and growth",
            "senior": "senior candidate - focus on leadership, strategy, and deep expertise",
            "unknown": "candidate with unknown experience level - assess as you go",
        }
        experience_context = experience_contexts.get(exp_level, experience_contexts["unknown"])
        
        # Use memory for conversation context if available
        if memory:
            conversation_context = memory.get_full_conversation_for_llm(
                max_turns=8, max_chars_per_turn=300
            )
            hypotheses_context = memory.get_hypotheses_for_llm(max_items=4)
            # Use memory's skill tracking
            assessed_skills = ", ".join(sorted(memory.assessed_skills)[:10]) if memory.assessed_skills else "None yet"
            unassessed = memory.unassessed_skills
            unassessed_skills = ", ".join(sorted(unassessed)[:5]) if unassessed else "All mentioned skills covered"
        else:
            conversation_context = self._build_conversation_context(state)
            assessed_skills = self._get_assessed_skills(state)
            unassessed_skills = self._get_unassessed_skills(state)
            hypotheses_context = "(None)"
        
        prompt = self.PLANNING_PROMPT.format(
            job_title=state.config.job_title,
            required_skills=required_skills,
            candidate_experience_level=exp_level,
            experience_context=experience_context,
            phase=state.current_phase.value,
            core_questions_asked=state.core_questions_asked,
            max_core_questions=max_core,
            assessed_skills=assessed_skills,
            unassessed_skills=unassessed_skills,
            current_topic=current_topic,
            followups_on_topic=followups_on_topic,
            max_followups=max_followups,
            red_flag_count=memory.red_flag_count if memory else len(state.red_flags),
            hypotheses_context=hypotheses_context,
            conversation_context=conversation_context,
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            # Parse next question
            next_q = response.get("next_question", {})
            q_type = next_q.get("question_type", "open_ended")
            try:
                question_type = QuestionType(q_type)
            except ValueError:
                question_type = QuestionType.OPEN_ENDED

            next_question = PlannedQuestion(
                question_text=next_q.get("question_text", "Please tell me more."),
                question_type=question_type,
                target_skill=next_q.get("target_skill"),
                priority=float(next_q.get("priority", 0.8)),
                rationale=next_q.get("rationale", ""),
            )

            # Parse phase transition
            should_transition = response.get("should_transition_phase", False)
            recommended_phase = None
            if should_transition and response.get("recommended_phase"):
                try:
                    recommended_phase = InterviewPhase(response["recommended_phase"])
                except ValueError:
                    should_transition = False

            # Parse wrap-up recommendation
            should_wrap_up = response.get("should_wrap_up", False)
            
            # Force wrap-up if we've hit the core question limit
            if state.core_questions_asked >= state.config.max_core_questions:
                should_wrap_up = True

            return QuestionPlan(
                next_question=next_question,
                backup_questions=[],
                should_transition_phase=should_transition,
                recommended_phase=recommended_phase,
                should_wrap_up=should_wrap_up,
            )

        except OllamaError as e:
            logger.error(f"LLM question planning failed: {e}")
            return self._fallback_plan(state)

    def _fallback_plan(self, state: InterviewState) -> QuestionPlan:
        """Fallback question plan when LLM fails."""
        phase = state.current_phase

        fallback_questions = {
            InterviewPhase.INTRODUCTION: PlannedQuestion(
                question_text="Tell me about yourself and your professional background.",
                question_type=QuestionType.OPEN_ENDED,
                target_skill=None,
                priority=1.0,
                rationale="Standard opening question",
            ),
            InterviewPhase.BACKGROUND: PlannedQuestion(
                question_text="What aspects of your previous experience are most relevant to this role?",
                question_type=QuestionType.OPEN_ENDED,
                target_skill=None,
                priority=0.9,
                rationale="Understand relevant experience",
            ),
            InterviewPhase.TECHNICAL: PlannedQuestion(
                question_text="Can you describe a challenging technical problem you solved recently?",
                question_type=QuestionType.TECHNICAL,
                target_skill="problem_solving",
                priority=0.9,
                rationale="Assess technical problem-solving ability",
            ),
            InterviewPhase.BEHAVIORAL: PlannedQuestion(
                question_text="Tell me about a time when you had to work under pressure.",
                question_type=QuestionType.BEHAVIORAL,
                target_skill="stress_management",
                priority=0.8,
                rationale="Assess stress handling capability",
            ),
            InterviewPhase.QUESTIONS: PlannedQuestion(
                question_text="What questions do you have about the role or our company?",
                question_type=QuestionType.OPEN_ENDED,
                target_skill=None,
                priority=1.0,
                rationale="Give candidate opportunity to ask questions",
            ),
            InterviewPhase.CLOSING: PlannedQuestion(
                question_text="Is there anything else you'd like to share before we wrap up?",
                question_type=QuestionType.CLOSING,
                target_skill=None,
                priority=1.0,
                rationale="Final opportunity for candidate",
            ),
        }

        next_question = fallback_questions.get(
            phase,
            PlannedQuestion(
                question_text="Please continue.",
                question_type=QuestionType.FOLLOW_UP,
                priority=0.5,
                rationale="Default follow-up",
            ),
        )

        return QuestionPlan(
            next_question=next_question,
            backup_questions=[],
            should_transition_phase=False,
        )

    async def process(self, state: InterviewState, input_text: str) -> str:
        """
        Process method for Agent protocol compatibility.

        Args:
            state: Current interview state.
            input_text: Input to process (used for context).

        Returns:
            The planned question text.
        """
        plan = await self.plan_next(state)
        return plan.next_question.question_text
