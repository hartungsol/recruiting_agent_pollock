"""Hypothesis layer for controlled flexible reasoning.

Hypotheses are *explicitly* non-factual, defeasible inferences (often analogical or
generalized) that can guide question selection and soft scoring.

They must never be treated as asserted facts unless separately confirmed by
direct evidence during the interview.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis."""

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class Hypothesis(BaseModel):
    """A defeasible, explicitly unconfirmed inference."""

    hypothesis_id: UUID = Field(default_factory=uuid4, description="Unique hypothesis id")
    statement: str = Field(..., description="The hypothesis statement (non-factual)")
    confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Soft confidence in hypothesis (not a factual probability)",
    )
    basis: list[str] = Field(
        default_factory=list,
        description="Direct evidence snippets / observations supporting the hypothesis",
    )
    defeaters: list[str] = Field(
        default_factory=list,
        description="Potential defeaters/undercutters/rebutters for this hypothesis",
    )
    related_skills: list[str] = Field(
        default_factory=list,
        description="Skills/topics related to the hypothesis",
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.PROPOSED,
        description="Current hypothesis status",
    )
    created_turn: int = Field(default=-1, description="Turn index when created")
    updated_turn: int = Field(default=-1, description="Turn index when last updated")
