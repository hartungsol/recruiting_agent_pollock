"""
Recruiter Q&A agent.

Handles questions from candidates about the role, company, and process,
using retrieval-augmented generation for accurate responses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from recruiting_agent_pollock.models.llm_client import LLMClient, Message, OllamaError
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState

if TYPE_CHECKING:
    from recruiting_agent_pollock.memory import InterviewMemory

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    """Response to a candidate question."""

    answer: str = Field(..., description="Answer to the candidate's question")
    sources: list[str] = Field(
        default_factory=list,
        description="Sources used to generate the answer",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer accuracy",
    )
    needs_human_review: bool = Field(
        default=False,
        description="Whether answer should be reviewed by human recruiter",
    )


class RecruiterQABase(ABC):
    """Abstract base class for recruiter Q&A agents."""

    @abstractmethod
    async def answer_question(
        self,
        state: InterviewState,
        question: str,
    ) -> QAResponse:
        """
        Answer a candidate's question about the role or company.

        Args:
            state: Current interview state for context.
            question: Candidate's question.

        Returns:
            Response with answer and metadata.
        """
        ...


class RecruiterQA(RecruiterQABase):
    """
    Retrieval-augmented Q&A agent for candidate questions.

    Uses vector search to retrieve relevant information about the
    role and company, then generates accurate responses.
    """

    QA_PROMPT = """You are a helpful recruiter answering a candidate's question during an interview.
Be professional, honest, and informative. If you don't know something, say so.

Job Information:
- Title: {job_title}
- Company: {company_name}
- Required skills: {required_skills}
- Description: {job_description}

Retrieved context (if available):
{retrieved_context}

Candidate's question:
"{question}"

Provide a helpful, accurate answer. If the question is outside your knowledge, acknowledge that and offer to have a human recruiter follow up.

Return a JSON object:
{{
    "answer": "<your response to the candidate>",
    "confidence": <0.0-1.0 how confident you are in accuracy>,
    "needs_human_review": <true if a human should verify this>,
    "sources": ["<source1>", "<source2>"]
}}

Only return valid JSON, no other text."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        # vector_store: VectorStore | None = None,  # Future: for RAG
    ) -> None:
        """
        Initialize the recruiter Q&A agent.

        Args:
            llm_client: LLM client for response generation. Creates default if None.
        """
        self._llm_client = llm_client or LLMClient()
        # self._vector_store = vector_store  # Future: for RAG

    async def _retrieve_context(self, question: str, state: InterviewState) -> str:
        """
        Retrieve relevant context for answering the question.
        
        TODO: Integrate with vector store for RAG.
        """
        # Future: Use vector store to retrieve relevant documents
        # For now, return job config as context
        config = state.config
        context_parts = []
        
        if hasattr(config, 'job_description') and config.job_description:
            context_parts.append(f"Job Description: {config.job_description}")
        
        if hasattr(config, 'company_info') and config.company_info:
            context_parts.append(f"Company Info: {config.company_info}")
        
        if hasattr(config, 'benefits') and config.benefits:
            context_parts.append(f"Benefits: {', '.join(config.benefits)}")
        
        return "\n".join(context_parts) if context_parts else "(No additional context available)"

    async def answer_question(
        self,
        state: InterviewState,
        question: str,
        memory: InterviewMemory | None = None,
    ) -> QAResponse:
        """
        Answer a candidate's question about the role or company.

        Args:
            state: Current interview state for context.
            question: Candidate's question.
            memory: Optional interview memory (not heavily used here but kept for consistency).

        Returns:
            Response with answer and metadata.
        """
        retrieved_context = await self._retrieve_context(question, state)
        required_skills = ", ".join(state.config.required_skills[:8]) if state.config.required_skills else "Not specified"
        
        # Sanitize question if memory available
        if memory:
            sanitized_question = memory._sanitize_for_llm(question, max_length=300)
        else:
            sanitized_question = question[:300] if len(question) > 300 else question
        
        prompt = self.QA_PROMPT.format(
            job_title=state.config.job_title,
            company_name=getattr(state.config, 'company_name', 'Our company'),
            required_skills=required_skills,
            job_description=getattr(state.config, 'job_description', 'Not provided'),
            retrieved_context=retrieved_context,
            question=sanitized_question,
        )

        try:
            response = await self._llm_client.chat_with_json(
                messages=[Message(role="user", content=prompt)],
            )

            answer = response.get("answer", "I'm sorry, I couldn't generate a proper response. Let me have a recruiter follow up with you.")
            confidence = float(response.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            return QAResponse(
                answer=answer,
                sources=response.get("sources", []),
                confidence=confidence,
                needs_human_review=response.get("needs_human_review", confidence < 0.7),
            )

        except OllamaError as e:
            logger.error(f"LLM Q&A failed: {e}")
            return self._fallback_response(state, question)

    def _fallback_response(self, state: InterviewState, question: str) -> QAResponse:
        """Fallback response when LLM fails."""
        job_title = state.config.job_title
        
        # Simple keyword-based responses
        q_lower = question.lower()
        
        if any(word in q_lower for word in ["salary", "pay", "compensation", "money"]):
            return QAResponse(
                answer="Compensation details are typically discussed later in the process with our HR team. They'll provide a comprehensive overview of the total package.",
                sources=[],
                confidence=0.8,
                needs_human_review=True,
            )
        
        if any(word in q_lower for word in ["remote", "work from home", "office", "location"]):
            return QAResponse(
                answer="I'd be happy to discuss our work arrangements. Let me make a note to have our recruiter provide you with detailed information about our workplace policies.",
                sources=[],
                confidence=0.6,
                needs_human_review=True,
            )
        
        if any(word in q_lower for word in ["team", "who", "report", "manager"]):
            return QAResponse(
                answer=f"The {job_title} role works closely with various teams. I'll ensure you get detailed information about the team structure.",
                sources=[],
                confidence=0.6,
                needs_human_review=True,
            )
        
        return QAResponse(
            answer=f"That's a great question about the {job_title} position. Let me make a note of it, and our recruiting team will provide you with detailed information.",
            sources=[],
            confidence=0.5,
            needs_human_review=True,
        )

    async def process(self, state: InterviewState, input_text: str) -> str:
        """
        Process method for Agent protocol compatibility.

        Args:
            state: Current interview state.
            input_text: Question from candidate.

        Returns:
            Answer text.
        """
        response = await self.answer_question(state, input_text)
        return response.answer
