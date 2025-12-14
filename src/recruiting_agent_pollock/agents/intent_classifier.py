"""
Intent classifier agent.

Classifies candidate responses to understand their intent and
guide the interview flow.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState

if TYPE_CHECKING:
    from recruiting_agent_pollock.memory import InterviewMemory

logger = logging.getLogger(__name__)


class CandidateIntent(str, Enum):
    """Classification of candidate response intent."""

    ANSWER_QUESTION = "answer_question"
    ASK_CLARIFICATION = "ask_clarification"
    ASK_ABOUT_ROLE = "ask_about_role"
    ASK_ABOUT_COMPANY = "ask_about_company"
    PROVIDE_EXAMPLE = "provide_example"
    EXPRESS_CONCERN = "express_concern"
    EXPRESS_UNCERTAINTY = "express_uncertainty"  # "Will that be an issue?", "I don't know X"
    REDIRECT_TOPIC = "redirect_topic"
    CONFIRM = "confirm"
    DECLINE = "decline"
    OTHER = "other"


class IntentClassification(BaseModel):
    """Result of intent classification."""

    primary_intent: CandidateIntent = Field(..., description="Primary detected intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    secondary_intents: list[CandidateIntent] = Field(
        default_factory=list,
        description="Secondary intents detected",
    )
    extracted_entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Named entities extracted from the response",
    )


class IntentClassifierBase(ABC):
    """Abstract base class for intent classifiers."""

    @abstractmethod
    async def classify(
        self,
        state: InterviewState,
        input_text: str,
    ) -> IntentClassification:
        """
        Classify the intent of a candidate's response.

        Args:
            state: Current interview state for context.
            input_text: Candidate's response to classify.

        Returns:
            Classification result with intent and confidence.
        """
        ...


class IntentClassifier(IntentClassifierBase):
    """
    LLM-based intent classifier.

    Uses the LLM to classify candidate responses and extract
    relevant entities for downstream processing.
    """

    CLASSIFICATION_PROMPT = """You are an intent classifier for a recruiting interview system.
Analyze the candidate's response and classify their intent.

Available intents:
- answer_question: Candidate is answering an interview question
- ask_clarification: Candidate is asking for clarification on a question
- ask_about_role: Candidate is asking about the job role
- ask_about_company: Candidate is asking about the company
- provide_example: Candidate is providing a specific example or story
- express_concern: Candidate is expressing a concern or hesitation about the role
- express_uncertainty: Candidate is admitting they don't know something, asking "will that be an issue?", or showing honest uncertainty about their skills
- redirect_topic: Candidate is trying to change the subject
- confirm: Candidate is confirming or agreeing
- decline: Candidate is declining or refusing
- other: None of the above

IMPORTANT: "express_uncertainty" is for HONEST admissions like:
- "I don't have experience with X, will that be a problem?"
- "I'm not familiar with that tool"
- "I haven't done that before, but I'm eager to learn"
- "Is that something I'd need to know?"
This is NOT a red flag - it shows self-awareness and honesty.

Current interview context:
- Phase: {phase}
- Job Title: {job_title}
- Recent conversation:
{conversation_context}

Candidate's response to classify:
"{input_text}"

Respond with a JSON object containing:
{{
    "primary_intent": "<intent_name>",
    "confidence": <0.0-1.0>,
    "secondary_intents": ["<intent>", ...],
    "extracted_entities": {{"skill": "...", "experience": "...", ...}}
}}

Only return valid JSON, no other text."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the intent classifier.

        Args:
            llm_client: LLM client for classification. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()

    def _build_conversation_context(self, state: InterviewState) -> str:
        """Build conversation context string from state."""
        history = state.conversation_history[-6:]  # Last 6 exchanges
        if not history:
            return "(No prior conversation)"
        
        lines = []
        for entry in history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:200]  # Truncate long messages
            lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    async def classify(
        self,
        state: InterviewState,
        input_text: str,
        memory: InterviewMemory | None = None,
    ) -> IntentClassification:
        """
        Classify the intent of a candidate's response.

        Args:
            state: Current interview state for context.
            input_text: Candidate's response to classify.
            memory: Optional interview memory for enhanced context.

        Returns:
            Classification result with intent and confidence.
        """
        # Use memory for conversation context if available
        if memory:
            conversation_context = memory.get_full_conversation_for_llm(
                max_turns=6, max_chars_per_turn=200
            )
        else:
            conversation_context = self._build_conversation_context(state)
        
        prompt = self.CLASSIFICATION_PROMPT.format(
            phase=state.current_phase.value,
            job_title=state.config.job_title,
            conversation_context=conversation_context,
            input_text=input_text,
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            # Parse the JSON response
            primary = response.get("primary_intent", "other")
            try:
                primary_intent = CandidateIntent(primary)
            except ValueError:
                logger.warning(f"Unknown intent '{primary}', defaulting to OTHER")
                primary_intent = CandidateIntent.OTHER

            confidence = float(response.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))

            secondary_raw = response.get("secondary_intents", [])
            secondary_intents = []
            for s in secondary_raw:
                try:
                    secondary_intents.append(CandidateIntent(s))
                except ValueError:
                    pass

            return IntentClassification(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary_intents,
                extracted_entities=response.get("extracted_entities", {}),
            )

        except OllamaError as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to simple heuristics
            return self._fallback_classify(input_text)

    def _fallback_classify(self, input_text: str) -> IntentClassification:
        """Simple heuristic fallback when LLM fails."""
        text_lower = input_text.lower().strip()
        
        if text_lower.endswith("?"):
            if any(word in text_lower for word in ["what", "how", "why", "when", "where"]):
                if any(word in text_lower for word in ["role", "position", "job", "responsibilities"]):
                    return IntentClassification(
                        primary_intent=CandidateIntent.ASK_ABOUT_ROLE,
                        confidence=0.6,
                    )
                if any(word in text_lower for word in ["company", "team", "culture", "organization"]):
                    return IntentClassification(
                        primary_intent=CandidateIntent.ASK_ABOUT_COMPANY,
                        confidence=0.6,
                    )
                return IntentClassification(
                    primary_intent=CandidateIntent.ASK_CLARIFICATION,
                    confidence=0.5,
                )
        
        if any(word in text_lower for word in ["yes", "sure", "absolutely", "definitely", "i agree"]):
            return IntentClassification(
                primary_intent=CandidateIntent.CONFIRM,
                confidence=0.6,
            )
        
        if any(word in text_lower for word in ["no", "not really", "i can't", "i don't"]):
            return IntentClassification(
                primary_intent=CandidateIntent.DECLINE,
                confidence=0.5,
            )
        
        return IntentClassification(
            primary_intent=CandidateIntent.ANSWER_QUESTION,
            confidence=0.5,
        )

    async def process(self, state: InterviewState, input_text: str) -> str:
        """
        Process method for Agent protocol compatibility.

        Args:
            state: Current interview state.
            input_text: Input to process.

        Returns:
            String representation of classification.
        """
        classification = await self.classify(state, input_text)
        return classification.primary_intent.value
