"""
Pydantic schemas for the orchestrator module.

Defines data models for interview configuration, turns, and results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _now_utc() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# Experience level type for experience-aware interviewing
ExperienceLevel = Literal["entry", "mid", "senior", "unknown"]


class InterviewPhase(str, Enum):
    """Phases of the interview process."""

    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    QUESTIONS = "questions"
    CLOSING = "closing"


class TurnRole(str, Enum):
    """Role of the speaker in a conversation turn."""

    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    SYSTEM = "system"


class JobDescription(BaseModel):
    """
    Normalized job description schema.
    
    Used to drive interview logic, knockout rules, and scoring criteria.
    """

    job_id: str = Field(..., description="Unique identifier for the job position")
    title: str = Field(..., description="Job title")
    company_name: str = Field(default="", description="Company name")
    raw_text: str = Field(default="", description="Original raw job description text")
    min_experience_years: float | None = Field(
        default=None,
        description="Minimum years of experience required",
    )
    max_violations_3y: int | None = Field(
        default=None,
        description="Maximum allowed violations in last 3 years",
    )
    home_time: str | None = Field(
        default=None,
        description="Home time policy (e.g., 'weekly', 'bi-weekly', 'regional')",
    )
    equipment: list[str] = Field(
        default_factory=list,
        description="Required equipment/certifications (e.g., ['tanker', 'hazmat', 'doubles'])",
    )
    knockout_rules: list[str] = Field(
        default_factory=list,
        description="Hard disqualification rules (e.g., ['DUI in last 5 years', 'no valid CDL'])",
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="Required skills for the position",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Preferred but not required skills",
    )
    preferred_criteria: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Categorized preferred criteria (e.g., {'experience': [...], 'certifications': [...]})",
    )
    benefits: list[str] = Field(
        default_factory=list,
        description="Job benefits",
    )
    salary_range: str = Field(default="", description="Salary range if specified")
    location: str = Field(default="", description="Job location")
    job_type: str = Field(default="full-time", description="Job type (full-time, part-time, contract)")
    created_at: datetime = Field(default_factory=_now_utc, description="When the job was created")
    updated_at: datetime = Field(default_factory=_now_utc, description="When the job was last updated")

    def to_interview_config(self) -> "InterviewConfig":
        """
        Convert this job description to an InterviewConfig for interviews.
        
        Returns:
            InterviewConfig populated from this job description.
        """
        return InterviewConfig(
            job_id=self.job_id,
            job_title=self.title,
            company_name=self.company_name,
            job_description=self.raw_text,
            experience_level=f"{self.min_experience_years}+ years" if self.min_experience_years else "Not specified",
            required_skills=self.required_skills,
            preferred_skills=self.preferred_skills,
            benefits=self.benefits,
        )


class InterviewConfig(BaseModel):
    """Configuration for an interview session."""

    job_id: str = Field(..., description="Identifier for the job position")
    job_title: str = Field(..., description="Title of the job position")
    company_name: str = Field(default="Our Company", description="Name of the company")
    job_description: str = Field(default="", description="Full job description")
    experience_level: str = Field(default="mid-level", description="Required experience level")
    company_info: str = Field(default="", description="Information about the company")
    benefits: list[str] = Field(default_factory=list, description="Job benefits")
    required_skills: list[str] = Field(default_factory=list, description="Required skills for the position")
    preferred_skills: list[str] = Field(default_factory=list, description="Preferred skills for the position")
    max_duration_minutes: int = Field(default=60, description="Maximum interview duration in minutes")
    phases: list[InterviewPhase] = Field(
        default_factory=lambda: list(InterviewPhase),
        description="Interview phases to include",
    )
    # Interview pacing controls - for short, focused interviews
    max_core_questions: int = Field(
        default=5,
        description="Maximum number of core questions before offering to continue",
    )
    max_followups_per_topic: int = Field(
        default=2,
        description="Maximum follow-up questions per topic/skill area",
    )
    offer_continuation: bool = Field(
        default=True,
        description="Whether to offer candidate option to continue after core questions",
    )


class CandidateProfile(BaseModel):
    """Profile information about a candidate."""

    candidate_id: UUID = Field(default_factory=uuid4, description="Unique candidate identifier")
    name: str = Field(..., description="Candidate's full name")
    email: str = Field(..., description="Candidate's email address")
    resume_text: str = Field(default="", description="Extracted text from candidate's resume")
    experience_level: ExperienceLevel = Field(
        default="unknown",
        description="Candidate's experience level (entry, mid, senior, unknown)",
    )
    years_experience: float | None = Field(
        default=None,
        description="Years of relevant experience if known",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional candidate metadata")


class InterviewTurn(BaseModel):
    """A single turn in the interview conversation."""

    turn_id: UUID = Field(default_factory=uuid4, description="Unique turn identifier")
    role: TurnRole = Field(..., description="Role of the speaker")
    content: str = Field(..., description="Content of the turn")
    timestamp: datetime = Field(default_factory=_now_utc, description="When the turn occurred")
    phase: InterviewPhase = Field(..., description="Current interview phase")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional turn metadata")


class RedFlag(BaseModel):
    """A potential red flag identified during the interview."""

    flag_id: UUID = Field(default_factory=uuid4, description="Unique flag identifier")
    category: str = Field(..., description="Category of the red flag")
    description: str = Field(..., description="Description of the concern")
    severity: float = Field(default=0.5, ge=0.0, le=1.0, description="Severity score (0-1)")
    evidence: str = Field(default="", description="Supporting evidence from interview")
    mitigating_factors: list[str] = Field(default_factory=list, description="Factors that mitigate this concern")
    defeaters: list[str] = Field(default_factory=list, description="Potential defeaters from reasoning")


class SkillAssessment(BaseModel):
    """Assessment of a candidate's skill."""

    skill_name: str = Field(..., description="Name of the skill assessed")
    score: float = Field(..., ge=0.0, le=1.0, description="Assessed proficiency (0-1)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the assessment (0-1)")
    evidence: str = Field(default="", description="Supporting evidence")


class InterviewResult(BaseModel):
    """Final result of an interview session."""

    interview_id: UUID = Field(default_factory=uuid4, description="Unique interview identifier")
    candidate: CandidateProfile = Field(..., description="Candidate profile")
    config: InterviewConfig = Field(..., description="Interview configuration used")
    turns: list[InterviewTurn] = Field(default_factory=list, description="All conversation turns")
    skill_assessments: list[SkillAssessment] = Field(default_factory=list, description="Skill assessments")
    red_flags: list[RedFlag] = Field(default_factory=list, description="Identified red flags")
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0, description="Overall candidate score")
    recommendation: str = Field(default="", description="Final recommendation")
    reasoning_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Trace of defeasible reasoning steps",
    )
    started_at: datetime = Field(default_factory=_now_utc, description="Interview start time")
    completed_at: datetime | None = Field(default=None, description="Interview completion time")
