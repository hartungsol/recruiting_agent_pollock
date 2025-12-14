"""
Interview memory module.

Provides a per-interview memory that serves as the single source of truth
for all context, extracted facts, and assessments during an interview session.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory entries."""

    # Conversation-level memories
    UTTERANCE = "utterance"  # Raw conversation turn
    PARSED_RESPONSE = "parsed_response"  # Parsed/normalized response
    INTENT = "intent"  # Classified intent

    # Extracted facts
    SKILL_MENTION = "skill_mention"  # Candidate mentioned a skill
    EXPERIENCE_FACT = "experience_fact"  # Work experience detail
    ACHIEVEMENT = "achievement"  # Accomplishment mentioned
    EDUCATION = "education"  # Education details
    TIME_REFERENCE = "time_reference"  # Temporal information

    # Assessments
    SKILL_ASSESSMENT = "skill_assessment"  # Agent's skill assessment
    RED_FLAG = "red_flag"  # Detected red flag
    REASONING_STEP = "reasoning_step"  # Defeasible reasoning step
    HYPOTHESIS = "hypothesis"  # Explicit non-factual hypothesis (defeasible)

    # Interview control
    PHASE_TRANSITION = "phase_transition"  # Phase change event
    TOPIC_CHANGE = "topic_change"  # Topic transition
    QUESTION_ASKED = "question_asked"  # Question tracking


class MemoryEntry(BaseModel):
    """A single entry in interview memory."""

    entry_id: UUID = Field(default_factory=uuid4, description="Unique entry ID")
    memory_type: MemoryType = Field(..., description="Type of memory entry")
    content: dict[str, Any] = Field(..., description="Entry content/payload")
    source: str = Field(default="", description="Source agent or component")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this entry was created",
    )
    turn_index: int = Field(
        default=-1,
        description="Conversation turn this relates to (-1 if not turn-specific)",
    )
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How relevant this entry is (for retrieval)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class SkillContext(BaseModel):
    """Aggregated context about a skill from memory."""

    skill_name: str = Field(..., description="Name of the skill")
    mentions: list[str] = Field(default_factory=list, description="Direct mentions")
    assessments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Assessment entries for this skill",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence snippets supporting skill claims",
    )
    last_discussed_turn: int = Field(
        default=-1,
        description="Last turn where this skill was discussed",
    )


class CandidateSnapshot(BaseModel):
    """Point-in-time snapshot of candidate understanding."""

    experience_level: str = Field(default="unknown", description="Inferred experience level")
    years_experience: float | None = Field(default=None, description="Years of experience")
    confirmed_skills: list[str] = Field(default_factory=list, description="Skills with evidence")
    claimed_skills: list[str] = Field(default_factory=list, description="Skills mentioned but not verified")
    skill_gaps: list[str] = Field(default_factory=list, description="Identified skill gaps")
    strengths: list[str] = Field(default_factory=list, description="Identified strengths")
    red_flag_count: int = Field(default=0, description="Number of red flags")
    overall_impression: str = Field(default="neutral", description="Current impression")


class InterviewMemory:
    """
    Per-interview memory serving as single source of truth.

    This class manages all contextual information for an interview session,
    providing unified access to conversation history, extracted facts,
    assessments, and derived insights.

    Key responsibilities:
    - Store and index all memory entries by type, turn, and relevance
    - Provide efficient retrieval for agent context building
    - Maintain running aggregations (skills, red flags, etc.)
    - Generate context summaries for LLM consumption
    """

    def __init__(self, interview_id: UUID | None = None) -> None:
        """
        Initialize interview memory.

        Args:
            interview_id: Optional interview ID (generates new if None).
        """
        self._interview_id = interview_id or uuid4()
        self._entries: list[MemoryEntry] = []
        self._turn_count: int = 0
        self._created_at = datetime.now(timezone.utc)

        # Indices for fast retrieval
        self._by_type: dict[MemoryType, list[MemoryEntry]] = {t: [] for t in MemoryType}
        self._by_turn: dict[int, list[MemoryEntry]] = {}
        self._by_skill: dict[str, list[MemoryEntry]] = {}

        # Running aggregations
        self._mentioned_skills: set[str] = set()
        self._assessed_skills: set[str] = set()
        self._topics_covered: list[str] = []
        self._current_topic: str | None = None
        self._experience_level: str = "unknown"
        self._red_flag_count: int = 0

        logger.debug(f"Initialized InterviewMemory: {self._interview_id}")

    @property
    def interview_id(self) -> UUID:
        """Get the interview ID."""
        return self._interview_id

    @property
    def turn_count(self) -> int:
        """Get the current turn count."""
        return self._turn_count

    @property
    def entry_count(self) -> int:
        """Get total number of memory entries."""
        return len(self._entries)

    @property
    def mentioned_skills(self) -> set[str]:
        """Get all skills mentioned by candidate."""
        return self._mentioned_skills.copy()

    @property
    def assessed_skills(self) -> set[str]:
        """Get all skills that have been assessed."""
        return self._assessed_skills.copy()

    @property
    def unassessed_skills(self) -> set[str]:
        """Get skills mentioned but not yet assessed."""
        return self._mentioned_skills - self._assessed_skills

    @property
    def topics_covered(self) -> list[str]:
        """Get list of topics covered in order."""
        return self._topics_covered.copy()

    @property
    def current_topic(self) -> str | None:
        """Get the current topic being discussed."""
        return self._current_topic

    @property
    def experience_level(self) -> str:
        """Get the inferred experience level."""
        return self._experience_level

    @property
    def red_flag_count(self) -> int:
        """Get the number of red flags detected."""
        return self._red_flag_count

    def add_entry(
        self,
        memory_type: MemoryType,
        content: dict[str, Any],
        source: str = "",
        turn_index: int | None = None,
        relevance_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Add a new memory entry.

        Args:
            memory_type: Type of memory entry.
            content: Entry content/payload.
            source: Source agent or component.
            turn_index: Conversation turn (uses current if None).
            relevance_score: Relevance for retrieval (0-1).
            metadata: Additional metadata.

        Returns:
            The created MemoryEntry.
        """
        entry = MemoryEntry(
            memory_type=memory_type,
            content=content,
            source=source,
            turn_index=turn_index if turn_index is not None else self._turn_count,
            relevance_score=relevance_score,
            metadata=metadata or {},
        )

        # Store in main list
        self._entries.append(entry)

        # Update indices
        self._by_type[memory_type].append(entry)

        if entry.turn_index not in self._by_turn:
            self._by_turn[entry.turn_index] = []
        self._by_turn[entry.turn_index].append(entry)

        # Update skill index if applicable
        skill_name = content.get("skill_name") or content.get("target_skill")
        if skill_name:
            skill_lower = skill_name.lower()
            if skill_lower not in self._by_skill:
                self._by_skill[skill_lower] = []
            self._by_skill[skill_lower].append(entry)

        # Update running aggregations
        self._update_aggregations(entry)

        logger.debug(f"Added memory entry: {memory_type.value} from {source}")
        return entry

    def _update_aggregations(self, entry: MemoryEntry) -> None:
        """Update running aggregations based on new entry."""
        content = entry.content

        if entry.memory_type == MemoryType.SKILL_MENTION:
            skill_name = content.get("skill_name", "")
            if skill_name:
                self._mentioned_skills.add(skill_name.lower())

        elif entry.memory_type == MemoryType.SKILL_ASSESSMENT:
            skill_name = content.get("skill_name", "")
            if skill_name:
                self._assessed_skills.add(skill_name.lower())

        elif entry.memory_type == MemoryType.RED_FLAG:
            self._red_flag_count += 1

        elif entry.memory_type == MemoryType.TOPIC_CHANGE:
            new_topic = content.get("topic", "")
            if new_topic:
                if self._current_topic and self._current_topic not in self._topics_covered:
                    self._topics_covered.append(self._current_topic)
                self._current_topic = new_topic

        elif entry.memory_type == MemoryType.EXPERIENCE_FACT:
            # Try to infer experience level from facts
            years = content.get("years")
            if years is not None:
                if years < 2:
                    self._experience_level = "entry"
                elif years < 5:
                    self._experience_level = "mid"
                else:
                    self._experience_level = "senior"

    def increment_turn(self) -> int:
        """
        Increment turn counter and return new turn index.

        Returns:
            The new turn index.
        """
        self._turn_count += 1
        return self._turn_count

    def set_experience_level(self, level: str) -> None:
        """
        Explicitly set the experience level.

        Args:
            level: Experience level (entry, mid, senior, unknown).
        """
        if level in ("entry", "mid", "senior", "unknown"):
            self._experience_level = level

    def set_current_topic(self, topic: str | None) -> None:
        """
        Set the current discussion topic.

        Args:
            topic: Current topic name, or None to clear.
        """
        if self._current_topic and self._current_topic != topic:
            if self._current_topic not in self._topics_covered:
                self._topics_covered.append(self._current_topic)

            # Record topic change
            self.add_entry(
                memory_type=MemoryType.TOPIC_CHANGE,
                content={"topic": topic, "previous_topic": self._current_topic},
                source="memory",
            )

        self._current_topic = topic

    # =========================================================================
    # Retrieval Methods
    # =========================================================================

    def get_entries_by_type(self, memory_type: MemoryType) -> list[MemoryEntry]:
        """
        Get all entries of a specific type.

        Args:
            memory_type: Type to filter by.

        Returns:
            List of matching entries.
        """
        return self._by_type[memory_type].copy()

    def get_entries_for_turn(self, turn_index: int) -> list[MemoryEntry]:
        """
        Get all entries for a specific turn.

        Args:
            turn_index: Turn index to filter by.

        Returns:
            List of matching entries.
        """
        return self._by_turn.get(turn_index, []).copy()

    def get_entries_for_skill(self, skill_name: str) -> list[MemoryEntry]:
        """
        Get all entries related to a specific skill.

        Args:
            skill_name: Skill name to filter by.

        Returns:
            List of matching entries.
        """
        return self._by_skill.get(skill_name.lower(), []).copy()

    def get_recent_entries(
        self,
        count: int = 10,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemoryEntry]:
        """
        Get the most recent entries, optionally filtered by type.

        Args:
            count: Maximum number of entries to return.
            memory_types: Optional list of types to include.

        Returns:
            List of recent entries (newest last).
        """
        if memory_types:
            filtered = [e for e in self._entries if e.memory_type in memory_types]
        else:
            filtered = self._entries

        return filtered[-count:]

    def get_skill_context(self, skill_name: str) -> SkillContext:
        """
        Get aggregated context for a specific skill.

        Args:
            skill_name: Skill to get context for.

        Returns:
            Aggregated skill context.
        """
        entries = self.get_entries_for_skill(skill_name)

        mentions = []
        assessments = []
        evidence = []
        last_turn = -1

        for entry in entries:
            if entry.memory_type == MemoryType.SKILL_MENTION:
                mentions.append(entry.content.get("context", ""))
            elif entry.memory_type == MemoryType.SKILL_ASSESSMENT:
                assessments.append(entry.content)
                evidence_text = entry.content.get("evidence", "")
                if evidence_text:
                    evidence.append(evidence_text)

            if entry.turn_index > last_turn:
                last_turn = entry.turn_index

        return SkillContext(
            skill_name=skill_name,
            mentions=mentions,
            assessments=assessments,
            evidence=evidence,
            last_discussed_turn=last_turn,
        )

    def get_candidate_snapshot(self) -> CandidateSnapshot:
        """
        Get a point-in-time snapshot of candidate understanding.

        Returns:
            Current candidate snapshot.
        """
        # Collect confirmed vs claimed skills
        confirmed = []
        claimed = []
        gaps = []

        for skill in self._mentioned_skills:
            entries = self.get_entries_for_skill(skill)
            has_assessment = any(
                e.memory_type == MemoryType.SKILL_ASSESSMENT for e in entries
            )
            if has_assessment:
                # Check assessment score
                assessments = [
                    e for e in entries if e.memory_type == MemoryType.SKILL_ASSESSMENT
                ]
                if assessments:
                    latest = assessments[-1]
                    score = latest.content.get("score", 0.5)
                    if score >= 0.6:
                        confirmed.append(skill)
                    elif score < 0.4:
                        gaps.append(skill)
                    else:
                        claimed.append(skill)
            else:
                claimed.append(skill)

        # Identify strengths from high-scoring assessments
        strengths = []
        for entry in self.get_entries_by_type(MemoryType.SKILL_ASSESSMENT):
            if entry.content.get("score", 0) >= 0.8:
                strengths.append(entry.content.get("skill_name", ""))

        return CandidateSnapshot(
            experience_level=self._experience_level,
            confirmed_skills=confirmed,
            claimed_skills=claimed,
            skill_gaps=gaps,
            strengths=list(set(strengths)),
            red_flag_count=self._red_flag_count,
        )

    # =========================================================================
    # Context Building for Agents
    # =========================================================================

    def get_conversation_summary(self, max_turns: int = 5) -> str:
        """
        Get a summarized view of recent conversation.

        Args:
            max_turns: Maximum turns to include.

        Returns:
            Formatted conversation summary string.
        """
        utterances = self.get_entries_by_type(MemoryType.UTTERANCE)
        recent = utterances[-max_turns * 2 :] if max_turns else utterances

        lines = []
        for entry in recent:
            role = entry.content.get("role", "unknown")
            content = entry.content.get("content", "")
            # Truncate long responses for summary
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"  {role}: {content}")

        return "\n".join(lines) if lines else "(No conversation yet)"

    def get_skills_summary(self) -> str:
        """
        Get a summary of skills discussed.

        Returns:
            Formatted skills summary string.
        """
        if not self._mentioned_skills and not self._assessed_skills:
            return "No skills discussed yet"

        lines = []
        if self._assessed_skills:
            lines.append(f"Assessed: {', '.join(sorted(self._assessed_skills))}")
        if self._mentioned_skills - self._assessed_skills:
            unassessed = self._mentioned_skills - self._assessed_skills
            lines.append(f"Mentioned but not assessed: {', '.join(sorted(unassessed))}")

        return "\n".join(lines)

    def get_red_flags_summary(self) -> str:
        """
        Get a summary of detected red flags.

        Returns:
            Formatted red flags summary string.
        """
        flags = self.get_entries_by_type(MemoryType.RED_FLAG)
        if not flags:
            return "No red flags detected"

        lines = []
        for flag in flags:
            category = flag.content.get("category", "unknown")
            description = flag.content.get("description", "")
            severity = flag.content.get("severity", 0.5)
            lines.append(f"- [{category}] {description} (severity: {severity:.1f})")

        return "\n".join(lines)

    def build_agent_context(
        self,
        agent_name: str,
        include_conversation: bool = True,
        include_skills: bool = True,
        include_red_flags: bool = True,
        max_conversation_turns: int = 5,
    ) -> dict[str, Any]:
        """
        Build a context dictionary for an agent.

        This provides a standardized way for agents to get relevant
        context from memory.

        Args:
            agent_name: Name of the requesting agent (for logging).
            include_conversation: Include conversation summary.
            include_skills: Include skills summary.
            include_red_flags: Include red flags summary.
            max_conversation_turns: Max turns in conversation summary.

        Returns:
            Context dictionary for agent consumption.
        """
        context: dict[str, Any] = {
            "interview_id": str(self._interview_id),
            "turn_count": self._turn_count,
            "experience_level": self._experience_level,
            "current_topic": self._current_topic,
            "topics_covered": self._topics_covered,
        }

        if include_conversation:
            context["conversation_summary"] = self.get_conversation_summary(
                max_turns=max_conversation_turns
            )

        if include_skills:
            context["skills_summary"] = self.get_skills_summary()
            context["mentioned_skills"] = list(self._mentioned_skills)
            context["assessed_skills"] = list(self._assessed_skills)

        if include_red_flags:
            context["red_flags_summary"] = self.get_red_flags_summary()
            context["red_flag_count"] = self._red_flag_count

        logger.debug(f"Built context for agent '{agent_name}': {len(context)} fields")
        return context

    # =========================================================================
    # Hypotheses (Controlled Flexible Thinking)
    # =========================================================================

    def add_or_update_hypothesis(
        self,
        *,
        statement: str,
        confidence: float,
        basis: list[str] | None = None,
        defeaters: list[str] | None = None,
        related_skills: list[str] | None = None,
        status: str = "proposed",
        source: str = "",
    ) -> MemoryEntry:
        """Add a hypothesis, or update an existing one with the same statement.

        Hypotheses are explicitly NOT facts and must only be used as soft signals.
        """
        statement_norm = (statement or "").strip()
        if not statement_norm:
            raise ValueError("Hypothesis statement must be non-empty")

        basis = basis or []
        defeaters = defeaters or []
        related_skills = related_skills or []

        existing = None
        for entry in self.get_entries_by_type(MemoryType.HYPOTHESIS):
            if (entry.content.get("statement") or "").strip().lower() == statement_norm.lower():
                existing = entry
                break

        if existing is None:
            return self.add_entry(
                memory_type=MemoryType.HYPOTHESIS,
                content={
                    "statement": statement_norm,
                    "confidence": float(max(0.0, min(1.0, confidence))),
                    "basis": basis,
                    "defeaters": defeaters,
                    "related_skills": related_skills,
                    "status": status,
                },
                source=source,
            )

        # Update existing entry in-place (keep as SSOT).
        existing.content["confidence"] = float(
            max(existing.content.get("confidence", 0.0), max(0.0, min(1.0, confidence)))
        )
        existing.content["status"] = status

        # Merge lists without duplicates.
        def _merge(key: str, new_items: list[str]) -> None:
            prev = existing.content.get(key, []) or []
            merged: list[str] = []
            seen: set[str] = set()
            for item in [*prev, *new_items]:
                norm = (item or "").strip()
                if not norm:
                    continue
                nl = norm.lower()
                if nl in seen:
                    continue
                merged.append(norm)
                seen.add(nl)
            existing.content[key] = merged

        _merge("basis", basis)
        _merge("defeaters", defeaters)
        _merge("related_skills", related_skills)

        return existing

    def set_hypothesis_status(
        self,
        *,
        statement: str,
        status: str,
        additional_basis: list[str] | None = None,
        additional_defeaters: list[str] | None = None,
        source: str = "",
    ) -> MemoryEntry:
        """Set status for an existing hypothesis (or create a minimal one).

        This keeps hypotheses as explicit, non-factual soft signals.
        """
        return self.add_or_update_hypothesis(
            statement=statement,
            confidence=float(0.0),
            basis=additional_basis or [],
            defeaters=additional_defeaters or [],
            related_skills=[],
            status=status,
            source=source,
        )

    def get_hypotheses_for_llm(self, max_items: int = 5) -> str:
        """Format hypotheses for LLM consumption, explicitly labeled as unconfirmed."""
        entries = self.get_entries_by_type(MemoryType.HYPOTHESIS)
        if not entries:
            return "(None)"

        # Prefer higher confidence first.
        sorted_entries = sorted(
            entries,
            key=lambda e: float(e.content.get("confidence", 0.0)),
            reverse=True,
        )
        lines: list[str] = []
        for entry in sorted_entries[:max_items]:
            stmt = self._sanitize_for_llm(entry.content.get("statement", ""), max_length=220)
            conf = float(entry.content.get("confidence", 0.0))
            status = self._sanitize_for_llm(entry.content.get("status", "proposed"), max_length=40)
            lines.append(f"- ({status}, conf {conf:.2f}) {stmt}")

            defeaters = entry.content.get("defeaters", []) or []
            if defeaters:
                d0 = self._sanitize_for_llm(str(defeaters[0]), max_length=180)
                lines.append(f"  defeater: {d0}")

        return "\n".join(lines) if lines else "(None)"

    # =========================================================================
    # LLM-Safe Context Building
    # =========================================================================

    @staticmethod
    def _sanitize_for_llm(text: str, max_length: int = 500) -> str:
        """
        Sanitize text for safe LLM consumption.

        Removes potential prompt injection patterns and truncates.

        Args:
            text: Raw text to sanitize.
            max_length: Maximum length to return.

        Returns:
            Sanitized text safe for LLM prompts.
        """
        if not text:
            return ""

        # Remove potential prompt injection patterns
        sanitized = text

        # Remove common instruction override patterns
        dangerous_patterns = [
            "ignore previous instructions",
            "ignore all instructions",
            "disregard previous",
            "forget everything",
            "new instructions:",
            "system:",
            "```json",  # Prevent JSON injection in non-JSON contexts
            "```python",
            "<|",  # Common special tokens
            "|>",
        ]
        text_lower = sanitized.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in text_lower:
                # Replace with safe placeholder
                import re
                sanitized = re.sub(
                    re.escape(pattern),
                    "[FILTERED]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."

        return sanitized.strip()

    def get_candidate_utterances(self, max_count: int = 10) -> list[str]:
        """
        Get recent candidate utterances for LLM context.

        Args:
            max_count: Maximum number of utterances to return.

        Returns:
            List of sanitized candidate utterances (most recent last).
        """
        utterances = self.get_entries_by_type(MemoryType.UTTERANCE)
        candidate_only = [
            e for e in utterances if e.content.get("role") == "candidate"
        ]
        recent = candidate_only[-max_count:]

        return [
            self._sanitize_for_llm(e.content.get("content", ""), max_length=400)
            for e in recent
        ]

    def get_full_conversation_for_llm(
        self,
        max_turns: int = 10,
        max_chars_per_turn: int = 300,
    ) -> str:
        """
        Get the full conversation formatted safely for LLM consumption.

        Args:
            max_turns: Maximum number of turns to include.
            max_chars_per_turn: Max characters per turn.

        Returns:
            Formatted conversation string safe for LLM prompts.
        """
        utterances = self.get_entries_by_type(MemoryType.UTTERANCE)
        recent = utterances[-max_turns:] if max_turns else utterances

        lines = []
        for entry in recent:
            role = entry.content.get("role", "unknown")
            content = self._sanitize_for_llm(
                entry.content.get("content", ""),
                max_length=max_chars_per_turn,
            )
            # Use consistent role labels
            role_label = "Interviewer" if role == "interviewer" else "Candidate"
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines) if lines else "(No conversation yet)"

    def get_key_facts_for_llm(self) -> str:
        """
        Get key extracted facts formatted for LLM context.

        Returns:
            Formatted facts string safe for LLM prompts.
        """
        lines = []

        # Experience facts
        exp_facts = self.get_entries_by_type(MemoryType.EXPERIENCE_FACT)
        if exp_facts:
            lines.append("Experience mentioned:")
            for fact in exp_facts[-5:]:  # Last 5 experience facts
                desc = self._sanitize_for_llm(
                    fact.content.get("description", ""),
                    max_length=150,
                )
                if desc:
                    lines.append(f"  - {desc}")

        # Achievements
        achievements = self.get_entries_by_type(MemoryType.ACHIEVEMENT)
        if achievements:
            lines.append("Achievements mentioned:")
            for ach in achievements[-3:]:
                desc = self._sanitize_for_llm(
                    ach.content.get("description", ""),
                    max_length=150,
                )
                if desc:
                    lines.append(f"  - {desc}")

        # Skills with context
        if self._mentioned_skills:
            lines.append(f"Skills mentioned: {', '.join(sorted(self._mentioned_skills)[:10])}")

        return "\n".join(lines) if lines else "No key facts extracted yet"

    def build_llm_context(
        self,
        agent_name: str,
        include_conversation: bool = True,
        include_facts: bool = True,
        include_assessments: bool = False,
        max_conversation_turns: int = 8,
    ) -> str:
        """
        Build a complete, sanitized context string for LLM agents.

        This is the primary method agents should use to get memory context.

        Args:
            agent_name: Name of requesting agent (for logging/debugging).
            include_conversation: Include conversation history.
            include_facts: Include extracted facts.
            include_assessments: Include skill assessments and red flags.
            max_conversation_turns: Max conversation turns to include.

        Returns:
            Formatted context string safe for LLM prompts.
        """
        sections = []

        # Candidate snapshot
        sections.append(f"Candidate Experience Level: {self._experience_level}")
        if self._current_topic:
            sections.append(f"Current Topic: {self._current_topic}")

        # Conversation
        if include_conversation:
            conv = self.get_full_conversation_for_llm(
                max_turns=max_conversation_turns,
            )
            sections.append(f"\nConversation:\n{conv}")

        # Facts
        if include_facts:
            facts = self.get_key_facts_for_llm()
            if facts and facts != "No key facts extracted yet":
                sections.append(f"\nKey Facts:\n{facts}")

        # Assessments (optional - may bias some agents)
        if include_assessments:
            if self._assessed_skills:
                sections.append(f"\nSkills assessed: {', '.join(sorted(self._assessed_skills)[:8])}")
            if self._red_flag_count > 0:
                sections.append(f"Red flags detected: {self._red_flag_count}")

        context = "\n".join(sections)
        logger.debug(
            f"Built LLM context for '{agent_name}': {len(context)} chars, "
            f"{self._turn_count} turns"
        )
        return context

    # =========================================================================
    # Memory Lifecycle Management
    # =========================================================================

    def clear(self) -> None:
        """
        Clear all memory entries and reset state.

        Call this at the end of an interview to free memory.
        This is the proper way to dispose of per-interview memory.
        """
        logger.info(f"Clearing InterviewMemory: {self._interview_id}")

        # Clear all entries
        self._entries.clear()

        # Clear indices
        for mem_type in MemoryType:
            self._by_type[mem_type].clear()
        self._by_turn.clear()
        self._by_skill.clear()

        # Reset aggregations
        self._mentioned_skills.clear()
        self._assessed_skills.clear()
        self._topics_covered.clear()
        self._current_topic = None
        self._red_flag_count = 0

        # Keep interview_id and turn_count for reference
        # but mark as cleared
        self._turn_count = 0

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory statistics.
        """
        return {
            "interview_id": str(self._interview_id),
            "total_entries": len(self._entries),
            "turn_count": self._turn_count,
            "entries_by_type": {
                t.value: len(self._by_type[t]) for t in MemoryType
            },
            "skills_mentioned": len(self._mentioned_skills),
            "skills_assessed": len(self._assessed_skills),
            "topics_covered": len(self._topics_covered),
            "red_flags": self._red_flag_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize memory to dictionary for persistence.

        Returns:
            Dictionary representation of memory.
        """
        return {
            "interview_id": str(self._interview_id),
            "turn_count": self._turn_count,
            "created_at": self._created_at.isoformat(),
            "experience_level": self._experience_level,
            "current_topic": self._current_topic,
            "topics_covered": self._topics_covered,
            "mentioned_skills": list(self._mentioned_skills),
            "assessed_skills": list(self._assessed_skills),
            "red_flag_count": self._red_flag_count,
            "entries": [
                {
                    "entry_id": str(e.entry_id),
                    "memory_type": e.memory_type.value,
                    "content": e.content,
                    "source": e.source,
                    "timestamp": e.timestamp.isoformat(),
                    "turn_index": e.turn_index,
                    "relevance_score": e.relevance_score,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterviewMemory:
        """
        Deserialize memory from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed InterviewMemory.
        """
        memory = cls(interview_id=UUID(data["interview_id"]))
        memory._turn_count = data.get("turn_count", 0)
        memory._created_at = datetime.fromisoformat(data["created_at"])
        memory._experience_level = data.get("experience_level", "unknown")
        memory._current_topic = data.get("current_topic")
        memory._topics_covered = data.get("topics_covered", [])
        memory._mentioned_skills = set(data.get("mentioned_skills", []))
        memory._assessed_skills = set(data.get("assessed_skills", []))
        memory._red_flag_count = data.get("red_flag_count", 0)

        for entry_data in data.get("entries", []):
            entry = MemoryEntry(
                entry_id=UUID(entry_data["entry_id"]),
                memory_type=MemoryType(entry_data["memory_type"]),
                content=entry_data["content"],
                source=entry_data.get("source", ""),
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                turn_index=entry_data.get("turn_index", -1),
                relevance_score=entry_data.get("relevance_score", 1.0),
                metadata=entry_data.get("metadata", {}),
            )
            memory._entries.append(entry)
            memory._by_type[entry.memory_type].append(entry)

            if entry.turn_index not in memory._by_turn:
                memory._by_turn[entry.turn_index] = []
            memory._by_turn[entry.turn_index].append(entry)

            # Rebuild skill index
            skill_name = entry.content.get("skill_name") or entry.content.get("target_skill")
            if skill_name:
                skill_lower = skill_name.lower()
                if skill_lower not in memory._by_skill:
                    memory._by_skill[skill_lower] = []
                memory._by_skill[skill_lower].append(entry)

        return memory

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"InterviewMemory(id={self._interview_id}, "
            f"turns={self._turn_count}, entries={len(self._entries)})"
        )
