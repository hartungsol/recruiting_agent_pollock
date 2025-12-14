"""Hypothesis generation and defeater attachment.

This component proposes *explicit* hypotheses from grounded observations.
It is intentionally conservative and produces only a small number of hypotheses.

Hypotheses are stored in `InterviewMemory` and used only as soft signals for
question selection and scoring guidance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from recruiting_agent_pollock.memory import InterviewMemory
from recruiting_agent_pollock.reasoning.oscar_adapter import OSCARAdapter


class HypothesisEngine:
    """Generate controlled hypotheses and attach defeaters."""

    def __init__(self, oscar_adapter: OSCARAdapter | None = None) -> None:
        self._oscar = oscar_adapter or OSCARAdapter()

    def _iter_required_skill_gaps(
        self,
        required_skills: Iterable[str],
        mentioned_skills: set[str],
    ) -> list[str]:
        gaps: list[str] = []
        for s in required_skills:
            if s and s.lower() not in mentioned_skills:
                gaps.append(s)
        return gaps

    def _detect_explicit_skill_negations(
        self,
        text: str,
        parsed_skills: list[str],
        required_skills: list[str],
    ) -> list[str]:
        """Detect simple patterns like "not familiar with X".

        Prefer parsed skills when available; otherwise fall back to detecting
        mentions of required skills directly in the raw text.
        """
        t = (text or "").lower()
        if not re.search(r"\b(not familiar|never used|don'?t know|haven'?t used|no experience)\b", t):
            return []

        parsed = [s for s in (parsed_skills or []) if s]
        if parsed:
            return parsed

        matched: list[str] = []
        for s in required_skills or []:
            s_clean = (s or "").strip()
            if not s_clean:
                continue
            if s_clean.lower() in t:
                matched.append(s_clean)
        return matched

    def _get_latest_skill_assessment(self, memory: InterviewMemory, skill: str) -> tuple[float | None, float | None]:
        """Return (score, confidence) for the latest assessment of a skill."""
        from recruiting_agent_pollock.memory import MemoryType

        if not skill:
            return None, None

        assessments = memory.get_entries_by_type(MemoryType.SKILL_ASSESSMENT)
        for entry in reversed(assessments):
            if (entry.content.get("skill_name") or "").lower() == skill.lower():
                try:
                    return float(entry.content.get("score", 0.5)), float(entry.content.get("confidence", 0.5))
                except (TypeError, ValueError):
                    return None, None
        return None, None

    async def _auto_resolve_existing(self, *, memory: InterviewMemory) -> None:
        """Auto-mark hypotheses confirmed/rejected when later evidence appears."""
        from recruiting_agent_pollock.memory import MemoryType

        for entry in memory.get_entries_by_type(MemoryType.HYPOTHESIS):
            statement = (entry.content.get("statement") or "").strip()
            if not statement:
                continue

            status = (entry.content.get("status") or "proposed").strip().lower()
            if status in {"confirmed", "rejected"}:
                continue

            related_skills = entry.content.get("related_skills", []) or []
            skill = related_skills[0] if related_skills else ""

            stmt_l = statement.lower()

            # If the hypothesis is purely "not mentioned yet", reject once mentioned.
            if "not mentioned" in stmt_l and skill and skill.lower() in memory.mentioned_skills:
                memory.set_hypothesis_status(
                    statement=statement,
                    status="rejected",
                    additional_defeaters=["Skill has since been mentioned; absence-of-mention no longer applies."],
                    source="hypothesis_engine",
                )
                continue

            score, confidence = self._get_latest_skill_assessment(memory, skill)

            # Confirm a gap-like hypothesis if we later have a low assessed score.
            if ("limited exposure" in stmt_l or "lack hands-on" in stmt_l) and score is not None and confidence is not None:
                if confidence >= 0.6 and score <= 0.4:
                    memory.set_hypothesis_status(
                        statement=statement,
                        status="confirmed",
                        additional_basis=[f"Later assessment for {skill} was low (score {score:.2f}, conf {confidence:.2f})."],
                        source="hypothesis_engine",
                    )
                    continue

                # Reject uncertainty-based gap if we later have a strong assessment.
                if confidence >= 0.6 and score >= 0.6 and "self-reported uncertainty" in stmt_l:
                    memory.set_hypothesis_status(
                        statement=statement,
                        status="rejected",
                        additional_defeaters=[f"Later assessment for {skill} indicates competence (score {score:.2f}, conf {confidence:.2f})."],
                        source="hypothesis_engine",
                    )

    async def update_hypotheses(
        self,
        *,
        memory: InterviewMemory,
        candidate_text: str,
        required_skills: list[str],
        parsed_skills: list[str] | None = None,
    ) -> None:
        """Update memory with new/updated hypotheses derived from a turn."""

        parsed_skills = parsed_skills or []

        # 1) Hypothesize potential required-skill gaps (low confidence).
        gaps = self._iter_required_skill_gaps(required_skills, memory.mentioned_skills)
        for skill in gaps[:3]:
            statement = f"Candidate may have limited exposure to {skill} (not mentioned yet)."
            basis = ["Required skill has not been mentioned so far."]
            defeaters = await self._oscar.propose_defeaters(
                hypothesis_statement=statement,
                basis=basis,
                experience_level=memory.experience_level,
            )
            memory.add_or_update_hypothesis(
                statement=statement,
                confidence=0.25,
                basis=basis,
                defeaters=defeaters,
                related_skills=[skill],
                status="proposed",
                source="hypothesis_engine",
            )

        # 2) If the candidate expresses unfamiliarity, propose a higher-confidence gap.
        negated = self._detect_explicit_skill_negations(candidate_text, parsed_skills, required_skills)
        for skill in negated[:2]:
            statement = f"Candidate may lack hands-on experience with {skill} (self-reported uncertainty)."
            basis = ["Candidate expressed uncertainty / lack of familiarity."]
            defeaters = await self._oscar.propose_defeaters(
                hypothesis_statement=statement,
                basis=basis,
                experience_level=memory.experience_level,
            )
            memory.add_or_update_hypothesis(
                statement=statement,
                confidence=0.55,
                basis=basis,
                defeaters=defeaters,
                related_skills=[skill],
                status="confirmed",
                source="hypothesis_engine",
            )

        # 3) Auto-resolve older hypotheses based on new evidence.
        await self._auto_resolve_existing(memory=memory)
