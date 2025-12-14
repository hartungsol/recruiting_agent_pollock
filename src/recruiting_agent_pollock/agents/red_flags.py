"""
Red flag detection agent.

Identifies potential concerns in candidate responses using
both pattern matching and defeasible reasoning.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState
from recruiting_agent_pollock.orchestrator.schemas import RedFlag

if TYPE_CHECKING:
    from recruiting_agent_pollock.memory import InterviewMemory

logger = logging.getLogger(__name__)


class RedFlagCategory(str, Enum):
    """Categories of potential red flags."""

    INCONSISTENCY = "inconsistency"
    SKILL_GAP = "skill_gap"
    COMMUNICATION = "communication"
    PROFESSIONALISM = "professionalism"
    CULTURAL_FIT = "cultural_fit"
    EXPERIENCE_MISMATCH = "experience_mismatch"
    MOTIVATION = "motivation"
    RELIABILITY = "reliability"
    ETHICS = "ethics"


class RedFlagAnalysis(BaseModel):
    """Analysis result for red flag detection."""

    detected_flags: list[RedFlag] = Field(
        default_factory=list,
        description="Red flags detected in this response",
    )
    reasoning_applied: bool = Field(
        default=False,
        description="Whether defeasible reasoning was applied",
    )
    reasoning_summary: str = Field(
        default="",
        description="Summary of reasoning process",
    )


class RedFlagDetectorBase(ABC):
    """Abstract base class for red flag detectors."""

    @abstractmethod
    async def analyze(
        self,
        state: InterviewState,
        input_text: str,
    ) -> RedFlagAnalysis:
        """
        Analyze a response for potential red flags.

        Args:
            state: Current interview state for context.
            input_text: Candidate response to analyze.

        Returns:
            Analysis result with any detected flags.
        """
        ...


class RedFlagDetector(RedFlagDetectorBase):
    """
    Red flag detector using LLM and defeasible reasoning.

    Identifies potential concerns in candidate responses and uses
    the OSCAR reasoner to evaluate whether flags are defeated by
    additional context or explanations.
    """

    ANALYSIS_PROMPT = """You are a red flag detector for a recruiting interview system.
Analyze the candidate's response for potential concerns, but be FAIR and EXPERIENCE-AWARE.

Job context:
- Position: {job_title}
- Required skills: {required_skills}
- Interview phase: {phase}

CANDIDATE CONTEXT:
- Experience level: {experience_level}
{experience_guidance}

Recent conversation:
{conversation_context}

Latest candidate response:
"{input_text}"

Red flag categories to consider:
- inconsistency: Contradictions with previous statements
- professionalism: Unprofessional language or attitude
- cultural_fit: Values misalignment
- experience_mismatch: Experience doesn't match claims
- motivation: Lack of genuine interest
- reliability: Signs of unreliability
- ethics: Ethical concerns

IMPORTANT EXPERIENCE-AWARENESS RULES:
1. Entry-level candidates admitting skill gaps is HONEST, not a red flag
2. "I don't know X but I'm willing to learn" is POSITIVE, not negative
3. Nervousness or uncertainty in juniors is NORMAL
4. Only flag skill_gap for entry-level if they lack FUNDAMENTAL basics
5. Mid/senior candidates should be held to higher standards

Analyze carefully and return a JSON object:
{{
    "detected_flags": [
        {{
            "category": "<category>",
            "description": "<specific concern>",
            "severity": "<low|medium|high>",
            "evidence": "<quote or observation>",
            "mitigating_factors": ["<factor1>", ...]
        }}
    ],
    "reasoning_summary": "<brief explanation of your analysis>",
    "overall_concern_level": "<none|low|medium|high>"
}}

Only return valid JSON, no other text."""

    CONSISTENCY_CHECK_PROMPT = """You are checking for consistency across an interview.
Review all the candidate's responses and identify any contradictions.

Conversation history:
{full_conversation}

Look for:
1. Timeline inconsistencies (conflicting dates or durations)
2. Skill claim contradictions (claiming expertise then showing gaps)
3. Experience contradictions (different versions of same story)
4. Attitude shifts (enthusiasm to disinterest)

Return a JSON object:
{{
    "inconsistencies": [
        {{
            "type": "<timeline|skill|experience|attitude>",
            "description": "<what's inconsistent>",
            "evidence": ["<quote1>", "<quote2>"],
            "severity": "<low|medium|high>"
        }}
    ],
    "summary": "<brief summary>"
}}

Only return valid JSON, no other text."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        # oscar_adapter: OSCARAdapter | None = None,  # Future: for defeasible reasoning
    ) -> None:
        """
        Initialize the red flag detector.

        Args:
            llm_client: LLM client for analysis. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()
        # self._oscar_adapter = oscar_adapter  # Future

    def _build_conversation_context(self, state: InterviewState) -> str:
        """Build conversation context string from state."""
        history = state.conversation_history[-10:]
        if not history:
            return "(No prior conversation)"
        
        lines = []
        for entry in history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:400]
            lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    def _build_full_conversation(self, state: InterviewState) -> str:
        """Build full conversation for consistency checking."""
        lines = []
        for entry in state.conversation_history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def analyze(
        self,
        state: InterviewState,
        input_text: str,
        memory: InterviewMemory | None = None,
    ) -> RedFlagAnalysis:
        """
        Analyze a response for potential red flags.

        Args:
            state: Current interview state for context.
            input_text: Candidate response to analyze.
            memory: Optional interview memory for enhanced context.

        Returns:
            Analysis result with any detected flags.
        """
        required_skills = ", ".join(state.config.required_skills[:8]) if state.config.required_skills else "Not specified"
        
        # Get experience level - prefer memory if available
        if memory:
            experience_level = memory.experience_level
        else:
            experience_level = getattr(state, 'experience_level', 'unknown')
        
        # Build experience-specific guidance
        if experience_level == "entry":
            experience_guidance = """- IMPORTANT: Entry-level candidates are NOT expected to know everything
- Honest acknowledgment of skill gaps is a POSITIVE signal, not a red flag
- Focus on learning potential, attitude, and foundational knowledge
- Do NOT flag "I don't know" or "I haven't used that yet" responses"""
        elif experience_level == "mid":
            experience_guidance = """- Mid-level candidates should have solid fundamentals
- Some gaps are acceptable if they show strong core skills
- Evaluate depth of experience in claimed areas"""
        elif experience_level == "senior":
            experience_guidance = """- Senior candidates should demonstrate deep expertise
- Evaluate leadership, mentorship, and strategic thinking
- Gaps in cutting-edge tools are less concerning than gaps in fundamentals"""
        else:
            experience_guidance = "- Experience level unknown; assess based on their stated background"
        
        # Use memory for conversation context if available
        if memory:
            conversation_context = memory.get_full_conversation_for_llm(
                max_turns=6, max_chars_per_turn=250
            )
            sanitized_input = memory._sanitize_for_llm(input_text, max_length=500)
        else:
            conversation_context = self._build_conversation_context(state)
            sanitized_input = input_text[:500] if len(input_text) > 500 else input_text
        
        prompt = self.ANALYSIS_PROMPT.format(
            job_title=state.config.job_title,
            required_skills=required_skills,
            phase=state.current_phase.value,
            experience_level=experience_level,
            experience_guidance=experience_guidance,
            conversation_context=conversation_context,
            input_text=sanitized_input,
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            detected_flags = []
            for flag_data in response.get("detected_flags", []):
                if not isinstance(flag_data, dict):
                    continue
                category = flag_data.get("category", "other")
                severity = flag_data.get("severity", "low")
                
                # Map severity to numeric
                severity_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
                severity_score = severity_map.get(severity, 0.5)
                
                detected_flags.append(RedFlag(
                    category=category,
                    description=flag_data.get("description", "Unspecified concern"),
                    severity=severity_score,
                    evidence=flag_data.get("evidence", ""),
                    mitigating_factors=flag_data.get("mitigating_factors", []),
                ))

            # Future: Apply defeasible reasoning via OSCAR adapter
            # to evaluate whether flags should be defeated by context
            reasoning_applied = False  # Will be True when OSCAR is integrated

            return RedFlagAnalysis(
                detected_flags=detected_flags,
                reasoning_applied=reasoning_applied,
                reasoning_summary=response.get("reasoning_summary", ""),
            )

        except OllamaError as e:
            logger.error(f"LLM red flag analysis failed: {e}")
            return RedFlagAnalysis(
                detected_flags=[],
                reasoning_applied=False,
                reasoning_summary=f"Analysis unavailable: {e}",
            )

    async def check_consistency(
        self,
        state: InterviewState,
    ) -> list[RedFlag]:
        """
        Check for inconsistencies across the entire interview.

        Args:
            state: Current interview state.

        Returns:
            List of inconsistency-related red flags.
        """
        if len(state.conversation_history) < 4:
            return []  # Not enough conversation to check

        prompt = self.CONSISTENCY_CHECK_PROMPT.format(
            full_conversation=self._build_full_conversation(state),
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            flags = []
            for inconsistency in response.get("inconsistencies", []):
                severity = inconsistency.get("severity", "low")
                severity_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
                
                flags.append(RedFlag(
                    category=RedFlagCategory.INCONSISTENCY.value,
                    description=inconsistency.get("description", "Inconsistency detected"),
                    severity=severity_map.get(severity, 0.5),
                    evidence="; ".join(inconsistency.get("evidence", [])),
                    mitigating_factors=[],
                ))
            
            return flags

        except OllamaError as e:
            logger.error(f"Consistency check failed: {e}")
            return []

    async def process(self, state: InterviewState, input_text: str) -> str:
        """
        Process method for Agent protocol compatibility.

        Args:
            state: Current interview state.
            input_text: Response to analyze.

        Returns:
            Summary of detected red flags.
        """
        analysis = await self.analyze(state, input_text)
        if analysis.detected_flags:
            flag_summaries = [f.description for f in analysis.detected_flags]
            return f"Detected {len(analysis.detected_flags)} potential concerns: {'; '.join(flag_summaries)}"
        return "No red flags detected"
