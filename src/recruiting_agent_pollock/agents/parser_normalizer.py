"""
Parser and normalizer agent.

Parses candidate responses to extract structured information and
normalizes them for downstream processing.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState

if TYPE_CHECKING:
    from recruiting_agent_pollock.memory import InterviewMemory

logger = logging.getLogger(__name__)


class ParsedResponse(BaseModel):
    """Structured representation of a parsed candidate response."""

    original_text: str = Field(..., description="Original response text")
    normalized_text: str = Field(..., description="Cleaned and normalized text")
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points extracted from the response",
    )
    mentioned_skills: list[str] = Field(
        default_factory=list,
        description="Skills mentioned in the response",
    )
    mentioned_experiences: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Work experiences mentioned",
    )
    mentioned_achievements: list[str] = Field(
        default_factory=list,
        description="Achievements or accomplishments mentioned",
    )
    time_references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Time periods mentioned (e.g., '3 years ago')",
    )
    sentiment: str = Field(
        default="neutral",
        description="Overall sentiment of the response",
    )
    confidence_indicators: list[str] = Field(
        default_factory=list,
        description="Phrases indicating confidence level",
    )


class ParserNormalizerBase(ABC):
    """Abstract base class for parser/normalizers."""

    @abstractmethod
    async def parse(
        self,
        state: InterviewState,
        input_text: str,
    ) -> ParsedResponse:
        """
        Parse and normalize a candidate response.

        Args:
            state: Current interview state for context.
            input_text: Raw candidate response.

        Returns:
            Structured parsed response.
        """
        ...


class ParserNormalizer(ParserNormalizerBase):
    """
    LLM-based response parser and normalizer.

    Extracts structured information from free-form candidate responses
    for use by other agents.
    """

    PARSING_PROMPT = """You are a response parser for a recruiting interview system.
Extract structured information from the candidate's response.

Job context:
- Position: {job_title}
- Required skills: {required_skills}
- Current interview phase: {phase}

Candidate's response:
"{input_text}"

Extract and return a JSON object with:
{{
    "normalized_text": "<cleaned version of the response>",
    "key_points": ["<main point 1>", "<main point 2>", ...],
    "mentioned_skills": ["<skill1>", "<skill2>", ...],
    "mentioned_experiences": [
        {{"role": "...", "company": "...", "duration": "...", "description": "..."}},
        ...
    ],
    "mentioned_achievements": ["<achievement1>", ...],
    "time_references": [
        {{"phrase": "3 years ago", "approximate_date": "2022"}},
        ...
    ],
    "sentiment": "<positive|neutral|negative|mixed>",
    "confidence_indicators": ["<phrases showing confidence or hesitation>", ...]
}}

Only return valid JSON, no other text."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the parser/normalizer.

        Args:
            llm_client: LLM client for parsing. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()

    @staticmethod
    def _clean_str_list(value: Any) -> list[str]:
        """Coerce a value into a list[str], dropping null/empty/non-string items."""
        if not value:
            return []
        if not isinstance(value, list):
            return []

        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            cleaned.append(s)
        return cleaned

    @staticmethod
    def _clean_dict_list(value: Any) -> list[dict[str, Any]]:
        """Coerce a value into a list[dict], dropping non-dicts."""
        if not value:
            return []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    async def parse(
        self,
        state: InterviewState,
        input_text: str,
        memory: InterviewMemory | None = None,
    ) -> ParsedResponse:
        """
        Parse and normalize a candidate response.

        Args:
            state: Current interview state for context.
            input_text: Raw candidate response.
            memory: Optional interview memory (used for context, not modified).

        Returns:
            Structured parsed response.
        """
        required_skills = ", ".join(state.config.required_skills[:10]) if state.config.required_skills else "Not specified"
        
        # Use memory to sanitize input if available
        if memory:
            sanitized_input = memory._sanitize_for_llm(input_text, max_length=1000)
        else:
            sanitized_input = input_text[:1000] if len(input_text) > 1000 else input_text
        
        prompt = self.PARSING_PROMPT.format(
            job_title=state.config.job_title,
            required_skills=required_skills,
            phase=state.current_phase.value,
            input_text=sanitized_input,
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            # LLM outputs are often noisy: guard against nulls in lists.
            normalized_text = response.get("normalized_text", input_text.strip())
            if not isinstance(normalized_text, str):
                normalized_text = input_text.strip()

            sentiment = response.get("sentiment", "neutral")
            if not isinstance(sentiment, str):
                sentiment = "neutral"

            return ParsedResponse(
                original_text=input_text,
                normalized_text=normalized_text,
                key_points=self._clean_str_list(response.get("key_points", [])),
                mentioned_skills=self._clean_str_list(response.get("mentioned_skills", [])),
                mentioned_experiences=self._clean_dict_list(response.get("mentioned_experiences", [])),
                mentioned_achievements=self._clean_str_list(response.get("mentioned_achievements", [])),
                time_references=self._clean_dict_list(response.get("time_references", [])),
                sentiment=sentiment,
                confidence_indicators=self._clean_str_list(response.get("confidence_indicators", [])),
            )

        except OllamaError as e:
            logger.error(f"LLM parsing failed: {e}")
            return self._fallback_parse(input_text)

    def _fallback_parse(self, input_text: str) -> ParsedResponse:
        """Simple heuristic fallback when LLM fails."""
        normalized = " ".join(input_text.split())
        
        # Extract simple key points (sentences)
        sentences = re.split(r'[.!?]+', normalized)
        key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
        
        # Simple sentiment detection
        positive_words = {"great", "excellent", "love", "enjoy", "successful", "achieved", "proud"}
        negative_words = {"difficult", "challenging", "problem", "issue", "failed", "struggled"}
        
        text_lower = input_text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Simple time reference extraction
        time_refs = []
        time_patterns = [
            r'(\d+)\s*years?\s*ago',
            r'in\s*(\d{4})',
            r'for\s*(\d+)\s*years?',
            r'since\s*(\d{4})',
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                time_refs.append({"phrase": match, "extracted": match})
        
        return ParsedResponse(
            original_text=input_text,
            normalized_text=normalized,
            key_points=key_points,
            mentioned_skills=[],
            mentioned_experiences=[],
            mentioned_achievements=[],
            time_references=time_refs,
            sentiment=sentiment,
            confidence_indicators=[],
        )

    async def process(self, state: InterviewState, input_text: str) -> str:
        """
        Process method for Agent protocol compatibility.

        Args:
            state: Current interview state.
            input_text: Input to process.

        Returns:
            Normalized text representation.
        """
        parsed = await self.parse(state, input_text)
        return parsed.normalized_text
