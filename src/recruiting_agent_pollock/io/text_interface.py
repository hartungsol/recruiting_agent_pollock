"""
Text-based interview interface.

Provides a command-line interface for conducting interviews
via text input/output.
"""

import os
from abc import ABC, abstractmethod

from recruiting_agent_pollock.models.llm_client import LLMClient
from recruiting_agent_pollock.orchestrator.interview_orchestrator import InterviewOrchestrator
from recruiting_agent_pollock.orchestrator.job_ingestion import JobIngestionService
from recruiting_agent_pollock.orchestrator.schemas import (
    CandidateProfile,
    InterviewConfig,
    InterviewResult,
    JobDescription,
)


class InterviewInterface(ABC):
    """Abstract base class for interview interfaces."""

    @abstractmethod
    async def run(self) -> None:
        """Run the interview interface."""
        ...

    @abstractmethod
    async def send_message(self, message: str) -> None:
        """
        Send a message to the user.

        Args:
            message: Message to display.
        """
        ...

    @abstractmethod
    async def receive_input(self) -> str:
        """
        Receive input from the user.

        Returns:
            User's input string.
        """
        ...


class TextInterface(InterviewInterface):
    """
    Command-line text interface for interviews.

    Provides a simple REPL for conducting interviews via terminal.
    """

    def __init__(
        self,
        orchestrator: InterviewOrchestrator,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the text interface.

        Args:
            orchestrator: Interview orchestrator to use.
            llm_client: LLM client for job parsing (uses orchestrator's if None).
        """
        self._orchestrator = orchestrator
        self._llm_client = llm_client
        self._job_ingestion = JobIngestionService(llm_client=llm_client)

    async def run(self) -> None:
        """Run the interactive interview session."""
        print("\n" + "=" * 60)
        print("Welcome to the Recruiting Agent Interview System")
        print("=" * 60 + "\n")

        # Get candidate info
        print("Please enter candidate information:")
        name = await self._get_input("Candidate name: ")
        email = await self._get_input("Candidate email: ")

        candidate = CandidateProfile(
            name=name,
            email=email,
        )

        # Get job description
        config = await self._get_job_config()

        # Start interview
        print("\n" + "-" * 60)
        print("Starting Interview")
        print("-" * 60 + "\n")

        introduction = await self._orchestrator.start_interview(config, candidate)
        await self.send_message(f"Interviewer: {introduction}")

        # Interview loop
        while self._orchestrator.is_active:
            candidate_input = await self.receive_input()

            if candidate_input.lower() in ("quit", "exit", "end"):
                print("\nEnding interview...")
                result = await self._orchestrator.end_interview()
                await self._display_result(result)
                break

            if candidate_input.strip():
                response = await self._orchestrator.process_candidate_input(candidate_input)
                await self.send_message(f"Interviewer: {response}")

    async def send_message(self, message: str) -> None:
        """
        Display a message to the terminal.

        Args:
            message: Message to display.
        """
        print(f"\n{message}\n")

    async def receive_input(self) -> str:
        """
        Get input from the terminal.

        Returns:
            User's input string.
        """
        return await self._get_input("You: ")

    async def _get_input(self, prompt: str) -> str:
        """
        Get input with a specific prompt.

        Args:
            prompt: Prompt to display.

        Returns:
            User's input.
        """
        # Using input() for simplicity; in production, could use aioconsole
        try:
            return input(prompt)
        except EOFError:
            return "exit"

    async def _get_job_config(self) -> InterviewConfig:
        """
        Get job configuration - either from a job description or simple input.

        Returns:
            InterviewConfig for the interview.
        """
        print("\nHow would you like to provide the job information?")
        print("  1. Enter job description text")
        print("  2. Load job description from file")
        print("  3. Quick setup (just job title)")

        choice = await self._get_input("\nChoice [1/2/3]: ")

        if choice == "1":
            return await self._get_job_from_text()
        elif choice == "2":
            return await self._get_job_from_file()
        else:
            # Default: quick setup
            return await self._get_job_simple()

    async def _get_job_simple(self) -> InterviewConfig:
        """Get a simple job config with just the title."""
        job_title = await self._get_input("Job title: ")
        return InterviewConfig(
            job_id="demo-job",
            job_title=job_title,
        )

    async def _get_job_from_text(self) -> InterviewConfig:
        """Parse job description from pasted text."""
        print("\nPaste the job description below.")
        print("Enter a blank line when done:\n")

        lines: list[str] = []
        while True:
            line = await self._get_input("")
            if not line.strip() and lines:
                break
            lines.append(line)

        raw_text = "\n".join(lines)

        if not raw_text.strip():
            print("No job description provided. Using simple setup.")
            return await self._get_job_simple()

        print("\nParsing job description...")
        try:
            job_description = await self._job_ingestion.ingest_from_text(raw_text)
            print(f"✓ Parsed: {job_description.title}")
            if job_description.required_skills:
                print(f"  Required skills: {', '.join(job_description.required_skills[:5])}")
            if job_description.knockout_rules:
                print(f"  Knockout rules: {len(job_description.knockout_rules)} rules")
            return job_description.to_interview_config()
        except Exception as e:
            print(f"Failed to parse job description: {e}")
            print("Falling back to simple setup.")
            return await self._get_job_simple()

    async def _get_job_from_file(self) -> InterviewConfig:
        """Load and parse job description from a file."""
        file_path = await self._get_input("File path: ")
        file_path = file_path.strip()

        if not file_path:
            print("No file path provided. Using simple setup.")
            return await self._get_job_simple()

        # Expand ~ and make absolute
        file_path = os.path.expanduser(file_path)
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            print("Using simple setup.")
            return await self._get_job_simple()

        print(f"\nLoading and parsing: {file_path}")
        try:
            job_description = await self._job_ingestion.ingest_from_file(file_path)
            print(f"✓ Parsed: {job_description.title}")
            if job_description.required_skills:
                print(f"  Required skills: {', '.join(job_description.required_skills[:5])}")
            if job_description.knockout_rules:
                print(f"  Knockout rules: {len(job_description.knockout_rules)} rules")
            return job_description.to_interview_config()
        except Exception as e:
            print(f"Failed to parse job description: {e}")
            print("Falling back to simple setup.")
            return await self._get_job_simple()

    async def _display_result(self, result: InterviewResult) -> None:
        """
        Display the interview result.

        Args:
            result: Interview result to display.
        """
        print("\n" + "=" * 60)
        print("Interview Summary")
        print("=" * 60)
        print(f"\nCandidate: {result.candidate.name}")
        print(f"Position: {result.config.job_title}")
        print(f"Duration: {result.started_at} to {result.completed_at}")
        print(f"Total turns: {len(result.turns)}")

        if result.overall_score is not None:
            print(f"\nOverall Score: {result.overall_score:.2f}")

        if result.skill_assessments:
            print("\nSkill Assessments:")
            for assessment in result.skill_assessments:
                print(f"  - {assessment.skill_name}: {assessment.score:.2f}")

        if result.red_flags:
            print("\nRed Flags:")
            for flag in result.red_flags:
                print(f"  - [{flag.category}] {flag.description} (severity: {flag.severity:.2f})")

        if result.recommendation:
            print(f"\nRecommendation: {result.recommendation}")

        print("\n" + "=" * 60)
