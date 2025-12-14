"""
Repository pattern for database operations.

Provides a clean abstraction over SQLAlchemy for CRUD operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recruiting_agent_pollock.db.models import (
    Base,
    CandidateModel,
    InterviewModel,
    InterviewTurnModel,
    JobModel,
)
from recruiting_agent_pollock.orchestrator.schemas import (
    CandidateProfile,
    InterviewResult,
    InterviewTurn,
    JobDescription,
)

T = TypeVar("T", bound=Base)


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the repository.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    @property
    @abstractmethod
    def _model_class(self) -> type[T]:
        """Get the model class for this repository."""
        ...

    async def get_by_id(self, entity_id: UUID) -> T | None:
        """
        Get an entity by its ID.

        Args:
            entity_id: The entity's UUID.

        Returns:
            The entity if found, None otherwise.
        """
        return await self._session.get(self._model_class, entity_id)

    async def create(self, entity: T) -> T:
        """
        Create a new entity.

        Args:
            entity: The entity to create.

        Returns:
            The created entity.
        """
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def update(self, entity: T) -> T:
        """
        Update an existing entity.

        Args:
            entity: The entity to update.

        Returns:
            The updated entity.
        """
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def delete(self, entity: T) -> None:
        """
        Delete an entity.

        Args:
            entity: The entity to delete.
        """
        await self._session.delete(entity)
        await self._session.flush()


class CandidateRepository(BaseRepository[CandidateModel]):
    """Repository for candidate operations."""

    @property
    def _model_class(self) -> type[CandidateModel]:
        """Get the model class."""
        return CandidateModel

    async def get_by_email(self, email: str) -> CandidateModel | None:
        """
        Get a candidate by email.

        Args:
            email: Candidate's email address.

        Returns:
            The candidate if found, None otherwise.
        """
        stmt = select(CandidateModel).where(CandidateModel.email == email)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_from_profile(self, profile: CandidateProfile) -> CandidateModel:
        """
        Create a candidate from a profile schema.

        Args:
            profile: Candidate profile data.

        Returns:
            The created candidate model.
        """
        candidate = CandidateModel(
            id=profile.candidate_id,
            name=profile.name,
            email=profile.email,
            resume_text=profile.resume_text or None,
            metadata_=profile.metadata,
        )
        return await self.create(candidate)

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[CandidateModel]:
        """
        List all candidates.

        Args:
            limit: Maximum number to return.
            offset: Number to skip.

        Returns:
            List of candidates.
        """
        stmt = select(CandidateModel).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class InterviewRepository(BaseRepository[InterviewModel]):
    """Repository for interview operations."""

    @property
    def _model_class(self) -> type[InterviewModel]:
        """Get the model class."""
        return InterviewModel

    async def get_by_candidate(
        self,
        candidate_id: UUID,
        limit: int = 100,
    ) -> list[InterviewModel]:
        """
        Get all interviews for a candidate.

        Args:
            candidate_id: Candidate's UUID.
            limit: Maximum number to return.

        Returns:
            List of interviews.
        """
        stmt = (
            select(InterviewModel)
            .where(InterviewModel.candidate_id == candidate_id)
            .order_by(InterviewModel.started_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def save_result(self, result: InterviewResult) -> InterviewModel:
        """
        Save an interview result.

        Args:
            result: Interview result to save.

        Returns:
            The created interview model.
        """
        # Create interview record
        interview = InterviewModel(
            id=result.interview_id,
            candidate_id=result.candidate.candidate_id,
            job_id=result.config.job_id,
            job_title=result.config.job_title,
            config=result.config.model_dump(),
            overall_score=result.overall_score,
            recommendation=result.recommendation or None,
            skill_assessments=[a.model_dump() for a in result.skill_assessments],
            red_flags=[f.model_dump() for f in result.red_flags],
            reasoning_trace=result.reasoning_trace,
            started_at=result.started_at,
            completed_at=result.completed_at,
        )

        # Create turn records
        for i, turn in enumerate(result.turns):
            turn_model = InterviewTurnModel(
                id=turn.turn_id,
                interview_id=result.interview_id,
                sequence=i,
                role=turn.role.value,
                content=turn.content,
                phase=turn.phase.value,
                metadata_=turn.metadata,
                timestamp=turn.timestamp,
            )
            interview.turns.append(turn_model)

        return await self.create(interview)

    async def get_recent(self, limit: int = 10) -> list[InterviewModel]:
        """
        Get recent interviews.

        Args:
            limit: Maximum number to return.

        Returns:
            List of recent interviews.
        """
        stmt = (
            select(InterviewModel)
            .order_by(InterviewModel.started_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class JobRepository(BaseRepository[JobModel]):
    """Repository for job description operations."""

    @property
    def _model_class(self) -> type[JobModel]:
        """Get the model class."""
        return JobModel

    async def get_by_external_id(self, external_id: str) -> JobModel | None:
        """
        Get a job by its external ID (job_id).

        Args:
            external_id: Job's external identifier.

        Returns:
            The job if found, None otherwise.
        """
        stmt = select(JobModel).where(JobModel.external_id == external_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_from_description(self, job: JobDescription) -> JobModel:
        """
        Create a job from a JobDescription schema.

        Args:
            job: JobDescription data.

        Returns:
            The created job model.
        """
        job_model = JobModel(
            external_id=job.job_id,
            title=job.title,
            company_name=job.company_name or None,
            description=job.raw_text[:1000] if job.raw_text else None,
            raw_text=job.raw_text or None,
            min_experience_years=job.min_experience_years,
            max_violations_3y=job.max_violations_3y,
            home_time=job.home_time,
            location=job.location or None,
            salary_range=job.salary_range or None,
            job_type=job.job_type,
            equipment=job.equipment,
            knockout_rules=job.knockout_rules,
            required_skills=job.required_skills,
            preferred_skills=job.preferred_skills,
            preferred_criteria=job.preferred_criteria,
            benefits=job.benefits,
        )
        return await self.create(job_model)

    async def update_from_description(
        self,
        job_model: JobModel,
        job: JobDescription,
    ) -> JobModel:
        """
        Update an existing job from a JobDescription schema.

        Args:
            job_model: Existing job model to update.
            job: New JobDescription data.

        Returns:
            The updated job model.
        """
        job_model.title = job.title
        job_model.company_name = job.company_name or None
        job_model.description = job.raw_text[:1000] if job.raw_text else ""
        job_model.raw_text = job.raw_text or None
        job_model.min_experience_years = job.min_experience_years
        job_model.max_violations_3y = job.max_violations_3y
        job_model.home_time = job.home_time
        job_model.location = job.location or None
        job_model.salary_range = job.salary_range or None
        job_model.job_type = job.job_type
        job_model.equipment = job.equipment
        job_model.knockout_rules = job.knockout_rules
        job_model.required_skills = job.required_skills
        job_model.preferred_skills = job.preferred_skills
        job_model.preferred_criteria = job.preferred_criteria
        job_model.benefits = job.benefits
        return await self.update(job_model)

    async def upsert_from_description(self, job: JobDescription) -> JobModel:
        """
        Create or update a job from a JobDescription.

        Args:
            job: JobDescription data.

        Returns:
            The created or updated job model.
        """
        existing = await self.get_by_external_id(job.job_id)
        if existing:
            return await self.update_from_description(existing, job)
        return await self.create_from_description(job)

    async def to_job_description(self, job_model: JobModel) -> JobDescription:
        """
        Convert a JobModel to a JobDescription schema.

        Args:
            job_model: Database job model.

        Returns:
            JobDescription schema.
        """
        return JobDescription(
            job_id=job_model.external_id,
            title=job_model.title,
            company_name=job_model.company_name or "",
            raw_text=job_model.raw_text or "",
            min_experience_years=job_model.min_experience_years,
            max_violations_3y=job_model.max_violations_3y,
            home_time=job_model.home_time,
            equipment=job_model.equipment,
            knockout_rules=job_model.knockout_rules,
            required_skills=job_model.required_skills,
            preferred_skills=job_model.preferred_skills,
            preferred_criteria=job_model.preferred_criteria,
            benefits=job_model.benefits,
            salary_range=job_model.salary_range or "",
            location=job_model.location or "",
            job_type=job_model.job_type,
        )

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[JobModel]:
        """
        List all jobs.

        Args:
            limit: Maximum number to return.
            offset: Number to skip.

        Returns:
            List of jobs.
        """
        stmt = (
            select(JobModel)
            .order_by(JobModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
