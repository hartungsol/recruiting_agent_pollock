"""
Interview orchestrator.

Coordinates the multi-agent interview process, managing state transitions,
agent invocations, and result aggregation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from recruiting_agent_pollock.agents import (
    IntentClassifier,
    ParserNormalizer,
    QuestionPlanner,
    RecruiterQA,
    RedFlagDetector,
    ScoringAgent,
)
from recruiting_agent_pollock.agents.intent_classifier import CandidateIntent
from recruiting_agent_pollock.memory import InterviewMemory, MemoryType
from recruiting_agent_pollock.models.llm_client import LLMClient
from recruiting_agent_pollock.orchestrator.interview_state import InterviewState
from recruiting_agent_pollock.orchestrator.schemas import (
    CandidateProfile,
    InterviewConfig,
    InterviewPhase,
    InterviewResult,
    TurnRole,
)
from recruiting_agent_pollock.reasoning.hypothesis_engine import HypothesisEngine

if TYPE_CHECKING:
    from recruiting_agent_pollock.agents.question_planner import QuestionPlan
    from recruiting_agent_pollock.agents.scoring import FinalScore


class InterviewOrchestrator:
    """
    Orchestrates the multi-agent interview process.

    Coordinates between various specialized agents (intent classification,
    question planning, red flag detection, etc.) to conduct a structured
    interview and generate assessments.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the interview orchestrator.

        Args:
            llm_client: LLM client for all agents. Creates default if None.
        """
        self._logger = logging.getLogger(__name__)
        self._current_state: InterviewState | None = None
        self._memory: InterviewMemory | None = None  # Per-interview memory
        
        # Initialize shared LLM client
        self._llm_client = llm_client or LLMClient()
        
        # Initialize all agents with shared LLM client
        self._intent_classifier = IntentClassifier(llm_client=self._llm_client)
        self._question_planner = QuestionPlanner(llm_client=self._llm_client)
        self._parser_normalizer = ParserNormalizer(llm_client=self._llm_client)
        self._recruiter_qa = RecruiterQA(llm_client=self._llm_client)
        self._red_flag_detector = RedFlagDetector(llm_client=self._llm_client)
        self._scoring_agent = ScoringAgent(llm_client=self._llm_client)

        # Controlled hypothesis layer (soft signals, defeasible)
        self._hypothesis_engine = HypothesisEngine()
        
        # Track questions asked per phase for phase transition logic
        self._questions_per_phase: dict[InterviewPhase, int] = {}

    @property
    def current_state(self) -> InterviewState | None:
        """Get the current interview state, if any."""
        return self._current_state

    @property
    def memory(self) -> InterviewMemory | None:
        """Get the current interview memory, if any."""
        return self._memory

    @property
    def is_active(self) -> bool:
        """Check if an interview is currently in progress."""
        return self._current_state is not None and not self._current_state.is_complete

    async def start_interview(
        self,
        config: InterviewConfig,
        candidate: CandidateProfile,
    ) -> str:
        """
        Start a new interview session.

        Args:
            config: Interview configuration.
            candidate: Candidate profile.

        Returns:
            Initial interviewer greeting/introduction.
        """
        self._logger.info(f"Starting interview for candidate: {candidate.name}")
        self._current_state = InterviewState(config=config, candidate=candidate)
        
        # Initialize per-interview memory
        self._memory = InterviewMemory(interview_id=self._current_state.interview_id)
        self._memory.set_experience_level(candidate.experience_level)

        # Generate introduction
        introduction = await self._generate_introduction()
        self._current_state.add_turn(TurnRole.INTERVIEWER, introduction)
        
        # Record in memory
        self._memory.increment_turn()
        self._memory.add_entry(
            memory_type=MemoryType.UTTERANCE,
            content={"role": "interviewer", "content": introduction},
            source="orchestrator",
        )

        return introduction

    async def process_candidate_input(self, input_text: str) -> str:
        """
        Process candidate input and generate the next interviewer response.

        Args:
            input_text: Candidate's response/input.

        Returns:
            Interviewer's next response.

        Raises:
            RuntimeError: If no interview is in progress.
        """
        if not self._current_state:
            raise RuntimeError("No active interview. Call start_interview first.")

        if self._current_state.is_complete:
            raise RuntimeError("Interview has already been completed.")

        # Record candidate's turn in state
        self._current_state.add_turn(TurnRole.CANDIDATE, input_text)
        
        # Record candidate utterance in memory
        if self._memory:
            self._memory.increment_turn()
            self._memory.add_entry(
                memory_type=MemoryType.UTTERANCE,
                content={"role": "candidate", "content": input_text},
                source="candidate",
            )

        # Try to infer experience level from early responses
        self._maybe_infer_experience(input_text)

        # Check if this is a response to continuation offer
        if self._current_state.offered_continuation and not self._current_state.candidate_wants_continue:
            wants_continue = self._check_continuation_response(input_text)
            self._current_state.set_candidate_wants_continue(wants_continue)
            if not wants_continue:
                # Candidate wants to wrap up
                response = self._generate_closing_message()
                self._record_interviewer_response(response)
                return response

        # 1. Classify intent
        intent_result = await self._intent_classifier.classify(
            self._current_state, input_text, memory=self._memory
        )
        self._logger.debug(f"Intent: {intent_result.primary_intent.value} (confidence: {intent_result.confidence:.2f})")
        
        # Record intent in memory
        if self._memory:
            self._memory.add_entry(
                memory_type=MemoryType.INTENT,
                content={
                    "intent": intent_result.primary_intent.value,
                    "confidence": intent_result.confidence,
                    "entities": intent_result.extracted_entities,
                },
                source="intent_classifier",
            )

        # 2. Parse and normalize response
        parsed = await self._parser_normalizer.parse(
            self._current_state, input_text, memory=self._memory
        )
        self._logger.debug(f"Parsed {len(parsed.key_points)} key points, {len(parsed.mentioned_skills)} skills")
        
        # Record parsed info in memory
        if self._memory:
            self._memory.add_entry(
                memory_type=MemoryType.PARSED_RESPONSE,
                content={
                    "key_points": parsed.key_points,
                    "sentiment": parsed.sentiment,
                },
                source="parser_normalizer",
            )
            # Record skill mentions
            for skill in parsed.mentioned_skills:
                self._memory.add_entry(
                    memory_type=MemoryType.SKILL_MENTION,
                    content={"skill_name": skill, "context": input_text[:200]},
                    source="parser_normalizer",
                )
            # Record experiences
            for exp in parsed.mentioned_experiences:
                self._memory.add_entry(
                    memory_type=MemoryType.EXPERIENCE_FACT,
                    content=exp,
                    source="parser_normalizer",
                )
            # Record achievements
            for ach in parsed.mentioned_achievements:
                self._memory.add_entry(
                    memory_type=MemoryType.ACHIEVEMENT,
                    content={"description": ach},
                    source="parser_normalizer",
                )

        # 2b. Update controlled hypotheses (non-factual; for planning/soft scoring only)
        if self._memory:
            await self._hypothesis_engine.update_hypotheses(
                memory=self._memory,
                candidate_text=input_text,
                required_skills=self._current_state.config.required_skills or [],
                parsed_skills=parsed.mentioned_skills,
            )

        # 3. Check for red flags (experience-aware)
        red_flag_analysis = await self._red_flag_detector.analyze(
            self._current_state, input_text, memory=self._memory
        )
        for flag in red_flag_analysis.detected_flags:
            self._current_state.add_red_flag(flag)
            self._logger.info(f"Red flag detected: {flag.category} - {flag.description}")
            # Record in memory
            if self._memory:
                self._memory.add_entry(
                    memory_type=MemoryType.RED_FLAG,
                    content={
                        "category": flag.category,
                        "description": flag.description,
                        "severity": flag.severity,
                    },
                    source="red_flag_detector",
                )

        # 4. Update scoring
        score_update = await self._scoring_agent.update_scores(
            self._current_state, input_text, memory=self._memory
        )
        for skill_assessment in score_update.skill_updates:
            self._current_state.add_skill_assessment(skill_assessment)
            # Record in memory
            if self._memory:
                self._memory.add_entry(
                    memory_type=MemoryType.SKILL_ASSESSMENT,
                    content={
                        "skill_name": skill_assessment.skill_name,
                        "score": skill_assessment.score,
                        "confidence": skill_assessment.confidence,
                        "evidence": skill_assessment.evidence,
                    },
                    source="scoring_agent",
                )
        self._logger.debug(f"Score delta: {score_update.overall_impression_delta:+.2f}")

        # 5. Generate response based on intent
        if intent_result.primary_intent in (
            CandidateIntent.ASK_ABOUT_ROLE,
            CandidateIntent.ASK_ABOUT_COMPANY,
            CandidateIntent.ASK_CLARIFICATION,
        ):
            # Use recruiter Q&A agent for candidate questions
            qa_response = await self._recruiter_qa.answer_question(
                self._current_state, input_text, memory=self._memory
            )
            response = qa_response.answer
        elif intent_result.primary_intent == CandidateIntent.EXPRESS_UNCERTAINTY:
            # Handle uncertainty gracefully - reassure and move on
            response = self._generate_uncertainty_response(input_text)
        else:
            # Plan next question
            question_plan = await self._question_planner.plan_next(
                self._current_state, memory=self._memory
            )
            
            # Check if we should offer continuation before more questions
            if self._current_state.should_offer_continuation():
                self._current_state.mark_continuation_offered()
                response = self._generate_continuation_offer()
            elif question_plan.should_wrap_up and not self._current_state.candidate_wants_continue:
                # Wrap up the interview
                response = self._generate_closing_message()
            else:
                # Check if we should transition phase
                if question_plan.should_transition_phase and question_plan.recommended_phase:
                    new_phase = self._current_state.advance_phase()
                    if new_phase:
                        self._logger.info(f"Transitioning to phase: {new_phase.value}")
                        # Record phase transition in memory
                        if self._memory:
                            self._memory.add_entry(
                                memory_type=MemoryType.PHASE_TRANSITION,
                                content={"new_phase": new_phase.value},
                                source="orchestrator",
                            )
                
                # Track topic and follow-ups
                target_skill = question_plan.next_question.target_skill
                if target_skill:
                    if question_plan.next_question.question_type.value == "follow_up":
                        self._current_state.increment_followup_for_topic(target_skill)
                    else:
                        # New topic
                        self._current_state.set_current_topic(target_skill)
                        self._current_state.increment_core_questions()
                        # Record topic change in memory
                        if self._memory:
                            self._memory.set_current_topic(target_skill)
                else:
                    self._current_state.increment_core_questions()
                
                response = question_plan.next_question.question_text
                
                # Record question in memory
                if self._memory:
                    self._memory.add_entry(
                        memory_type=MemoryType.QUESTION_ASKED,
                        content={
                            "question": response,
                            "type": question_plan.next_question.question_type.value,
                            "target_skill": target_skill,
                        },
                        source="question_planner",
                    )

        # Record interviewer's response
        self._record_interviewer_response(response)

        return response

    def record_interviewer_input(self, input_text: str, *, source: str = "human_interviewer") -> None:
        """Record a human interviewer's utterance without invoking any agents.

        This is used for voice (or text) scenarios where a human interviewer asks
        a question or adds context, and we want the orchestrator/agents to see it
        as part of the transcript for subsequent candidate turns.
        """
        if not self._current_state:
            raise RuntimeError("No active interview. Call start_interview first.")
        if self._current_state.is_complete:
            raise RuntimeError("Interview has already been completed.")

        text = (input_text or "").strip()
        if not text:
            return

        self._record_interviewer_response(text, source=source)

    def _record_interviewer_response(self, response: str, *, source: str = "orchestrator") -> None:
        """Record interviewer response in state and memory."""
        if not self._current_state:
            return
            
        self._current_state.add_turn(TurnRole.INTERVIEWER, response)
        
        # Track questions per phase
        phase = self._current_state.current_phase
        self._questions_per_phase[phase] = self._questions_per_phase.get(phase, 0) + 1
        
        # Record in memory
        if self._memory:
            self._memory.increment_turn()
            self._memory.add_entry(
                memory_type=MemoryType.UTTERANCE,
                content={"role": "interviewer", "content": response},
                source=source,
            )

    async def end_interview(self) -> InterviewResult:
        """
        End the current interview and generate final results.

        Returns:
            The complete interview result with assessments.

        Raises:
            RuntimeError: If no interview is in progress.
        """
        if not self._current_state:
            raise RuntimeError("No active interview to end.")

        self._logger.info(f"Ending interview: {self._current_state.interview_id}")
        
        # Log memory stats before clearing
        if self._memory:
            stats = self._memory.get_memory_stats()
            self._logger.debug(f"Interview memory stats: {stats}")

        # Check for consistency issues across the interview
        consistency_flags = await self._red_flag_detector.check_consistency(
            self._current_state
        )
        for flag in consistency_flags:
            self._current_state.add_red_flag(flag)

        # Generate final assessment using scoring agent
        final_score = await self._scoring_agent.calculate_final_score(
            self._current_state
        )

        result = self._current_state.complete(
            overall_score=final_score.overall_score,
            recommendation=final_score.recommendation,
        )

        # Clear per-interview memory (short-term memory lifecycle)
        if self._memory:
            self._memory.clear()
            self._memory = None
            self._logger.debug("Interview memory cleared")

        self._current_state = None
        self._questions_per_phase = {}
        return result

    async def _generate_introduction(self) -> str:
        """
        Generate the interview introduction.

        Returns:
            Introduction text.
        """
        if not self._current_state:
            return ""

        candidate_name = self._current_state.candidate.name
        job_title = self._current_state.config.job_title
        company_name = self._current_state.config.company_name

        return (
            f"Hello {candidate_name}, thank you for joining us today. "
            f"I'm excited to discuss the {job_title} position at {company_name} with you. "
            "Let's start by having you tell me a bit about yourself and your background."
        )

    def _generate_continuation_offer(self) -> str:
        """
        Generate the offer to continue or wrap up.
        
        Returns:
            Message offering candidate the choice to continue.
        """
        return (
            "Thank you for your responses so far - I have a good sense of your background. "
            "We can wrap up here, or if you'd like to share more about your experience or "
            "have questions about the role, I'm happy to continue. Would you like to continue, "
            "or shall we wrap up?"
        )

    def _generate_uncertainty_response(self, input_text: str) -> str:
        """
        Generate a graceful response to candidate expressing uncertainty or asking if gaps are an issue.
        
        Args:
            input_text: The candidate's uncertain response.
            
        Returns:
            Reassuring response that moves the interview forward.
        """
        text_lower = input_text.lower()
        
        # Check for "will that be an issue?" type questions
        if any(phrase in text_lower for phrase in ["be an issue", "be a problem", "deal breaker", "disqualify"]):
            return (
                "I appreciate your honesty - that's actually a positive trait we value. "
                "We're looking at the whole picture, not just individual skills. "
                "What matters most is your ability to learn and grow. "
                "Let's continue - tell me about a time you had to learn something new quickly."
            )
        
        # Check for "I don't know X" type statements
        if any(phrase in text_lower for phrase in ["don't know", "not familiar", "haven't used", "never done", "no experience"]):
            if self._current_state and self._current_state.is_entry_level():
                return (
                    "That's completely understandable for someone at your career stage. "
                    "We expect to train on specific tools. What I'm more interested in is - "
                    "how do you typically approach learning something new?"
                )
            else:
                return (
                    "I appreciate you being upfront about that. No one knows everything, "
                    "and we value honesty. Can you tell me about a similar skill or tool "
                    "you have used that might transfer?"
                )
        
        # Default reassuring response
        return (
            "I appreciate your candor - that self-awareness is valuable. "
            "Let's move on. What would you say is your strongest relevant skill for this role?"
        )

    def _generate_closing_message(self) -> str:
        """
        Generate the closing message.
        
        Returns:
            Closing message for the interview.
        """
        if not self._current_state:
            return "Thank you for your time today."
        
        # Prevent duplicate closing messages
        if self._current_state.closing_message_sent:
            return "Is there anything else you'd like to add before we finish?"
        
        self._current_state.mark_closing_sent()
        
        candidate_name = self._current_state.candidate.name
        job_title = self._current_state.config.job_title
        
        return (
            f"Thank you so much for speaking with me today, {candidate_name}. "
            f"I've really enjoyed learning about your background and experience. "
            f"We'll be reviewing all candidates for the {job_title} position and "
            "will be in touch soon with next steps. Do you have any final questions for me?"
        )

    def _check_continuation_response(self, input_text: str) -> bool:
        """
        Check if the candidate wants to continue based on their response.
        
        Args:
            input_text: Candidate's response to continuation offer.
            
        Returns:
            True if candidate wants to continue, False to wrap up.
        """
        text_lower = input_text.lower().strip()
        
        # Check for explicit "continue" signals
        continue_signals = [
            "continue", "yes", "more", "keep going", "go on", 
            "i'd like to", "sure", "let's continue", "please continue",
            "i have questions", "i want to", "happy to"
        ]
        
        # Check for explicit "wrap up" signals
        wrap_signals = [
            "wrap up", "no", "that's all", "i'm good", "we can stop",
            "let's finish", "done", "nothing else", "no questions",
            "all set", "good for now"
        ]
        
        for signal in wrap_signals:
            if signal in text_lower:
                return False
        
        for signal in continue_signals:
            if signal in text_lower:
                return True
        
        # Default: if they ask a question, they want to continue
        if "?" in input_text:
            return True
        
        # Default to wrapping up if unclear
        return False

    def _infer_experience_level(self, input_text: str) -> str | None:
        """
        Infer candidate experience level from their response.
        
        Uses simple heuristics to detect entry-level vs experienced candidates.
        
        Args:
            input_text: Candidate's response text.
            
        Returns:
            Inferred experience level or None if unclear.
        """
        text_lower = input_text.lower()
        
        # Entry-level indicators
        entry_patterns = [
            r"no\s*(real\s*)?experience",
            r"just\s*(graduated|finished|completed)",
            r"recent\s*graduate",
            r"only\s*(classes|school|coursework|bootcamp)",
            r"first\s*(job|position|role)",
            r"new\s*to\s*(the\s*)?(field|industry|qa|testing)",
            r"haven't\s*(worked|had\s*a\s*job)",
            r"looking\s*for\s*(my\s*)?first",
            r"entry[\s-]*level",
            r"no\s*professional\s*experience",
            r"internship\s*only",
            r"student",
        ]
        
        for pattern in entry_patterns:
            if re.search(pattern, text_lower):
                return "entry"
        
        # Senior indicators
        senior_patterns = [
            r"(\d+)\+?\s*years?\s*(of\s*)?(experience|in)",
            r"led\s*(a\s*)?(team|department|group)",
            r"senior\s*(position|role|engineer|tester|qa)",
            r"managed\s*(a\s*)?(team|project|department)",
            r"architect",
            r"principal",
            r"staff\s*engineer",
            r"director",
        ]
        
        for pattern in senior_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Check if years mentioned
                years_match = re.search(r"(\d+)\+?\s*years?", text_lower)
                if years_match:
                    years = int(years_match.group(1))
                    if years >= 7:
                        return "senior"
                    elif years >= 2:
                        return "mid"
                    else:
                        return "entry"
                # Leadership/senior title mentioned
                if any(term in text_lower for term in ["led", "managed", "senior", "principal", "director", "architect"]):
                    return "senior"
        
        # Mid-level indicators
        mid_patterns = [
            r"few\s*years",
            r"couple\s*(of\s*)?years",
            r"some\s*experience",
            r"worked\s*(at|for|with)\s*\w+",
            r"previous\s*(role|job|position)",
        ]
        
        for pattern in mid_patterns:
            if re.search(pattern, text_lower):
                return "mid"
        
        return None

    def _maybe_infer_experience(self, input_text: str) -> None:
        """
        Attempt to infer experience level if not already known.
        
        Only runs in the first few turns of the interview.
        
        Args:
            input_text: Candidate's response.
        """
        if not self._current_state:
            return
        
        # Only infer if not already known and early in interview
        if self._current_state.experience_inferred:
            return
        
        if len(self._current_state.turns) > 6:  # First 3 exchanges
            return
        
        inferred = self._infer_experience_level(input_text)
        if inferred:
            self._current_state.set_experience_level(inferred)
            # Sync to memory as well
            if self._memory:
                self._memory.set_experience_level(inferred)
            self._logger.info(f"Inferred experience level: {inferred}")

    def should_advance_phase(self) -> bool:
        """
        Determine if the interview should advance to the next phase.

        Returns:
            True if phase should advance.
        """
        if not self._current_state:
            return False
        
        phase = self._current_state.current_phase
        questions_asked = self._questions_per_phase.get(phase, 0)
        
        # Simple heuristic: advance after 3-5 questions per phase
        min_questions = 2 if phase == InterviewPhase.INTRODUCTION else 3
        return questions_asked >= min_questions

    async def advance_phase(self) -> InterviewPhase | None:
        """
        Advance to the next interview phase.

        Returns:
            The new phase, or None if no more phases.
        """
        if not self._current_state:
            return None

        new_phase = self._current_state.advance_phase()
        if new_phase:
            self._logger.info(f"Advancing to phase: {new_phase.value}")
        return new_phase
