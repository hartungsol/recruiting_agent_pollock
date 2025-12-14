"""
SQLAlchemy models for database persistence.

Defines the database schema for interviews, candidates, and related data.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class CandidateModel(Base):
    """Database model for candidates."""

    __tablename__ = "candidates"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    resume_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    interviews: Mapped[list["InterviewModel"]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
    )


class InterviewModel(Base):
    """Database model for interviews."""

    __tablename__ = "interviews"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    candidate_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("candidates.id"),
        nullable=False,
    )
    job_id: Mapped[str] = mapped_column(String(255), nullable=False)
    job_title: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    skill_assessments: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    red_flags: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    reasoning_trace: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    candidate: Mapped["CandidateModel"] = relationship(back_populates="interviews")
    turns: Mapped[list["InterviewTurnModel"]] = relationship(
        back_populates="interview",
        cascade="all, delete-orphan",
        order_by="InterviewTurnModel.sequence",
    )


class InterviewTurnModel(Base):
    """Database model for interview conversation turns."""

    __tablename__ = "interview_turns"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    interview_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interviews.id"),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    phase: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    interview: Mapped["InterviewModel"] = relationship(back_populates="turns")


class JobModel(Base):
    """Database model for job positions."""

    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Experience and violation requirements
    min_experience_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_violations_3y: Mapped[int | None] = mapped_column(nullable=True)
    
    # Job specifics
    home_time: Mapped[str | None] = mapped_column(String(100), nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    salary_range: Mapped[str | None] = mapped_column(String(100), nullable=True)
    job_type: Mapped[str] = mapped_column(String(50), default="full-time", nullable=False)
    
    # JSON fields for lists and complex data
    equipment: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    knockout_rules: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    required_skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    preferred_skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    preferred_criteria: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict, nullable=False)
    benefits: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
