"""
Main entry point for the Recruiting Agent Pollock application.
"""

import argparse
import asyncio
import logging
import sys

from recruiting_agent_pollock.config import get_settings
from recruiting_agent_pollock.io.text_interface import TextInterface
from recruiting_agent_pollock.models.llm_client import LLMClient
from recruiting_agent_pollock.orchestrator.interview_orchestrator import InterviewOrchestrator


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def run_interview(argv: list[str] | None = None) -> None:
    """
    Run an interactive interview session.

    This is the main async entry point that initializes all components
    and runs the interview loop.
    """
    settings = get_settings()
    logger = logging.getLogger(__name__)

    logger.info("Initializing Recruiting Agent Pollock...")
    logger.debug(f"Using LLM model: {settings.llm_model_name}")

    # Initialize LLM client with settings
    llm_client = LLMClient(
        model=settings.llm_model_name,
        timeout=settings.llm_timeout,
    )

    # Initialize orchestrator with LLM client
    orchestrator = InterviewOrchestrator(llm_client=llm_client)

    parser = argparse.ArgumentParser(prog="recruiting_agent_pollock")
    parser.add_argument(
        "--mode",
        choices=["text", "voice"],
        default="text",
        help="Run in text or voice mode",
    )
    args = parser.parse_args(argv)

    if args.mode == "voice":
        # Lazy import so text mode doesn't require optional voice deps.
        from recruiting_agent_pollock.io.voice_interface import VoiceInterface

        interface = VoiceInterface(orchestrator, llm_client=llm_client)
    else:
        interface = TextInterface(orchestrator, llm_client=llm_client)

    logger.info("Starting interview session...")
    await interface.run()


def main() -> None:
    """Main entry point for the application."""
    setup_logging()

    try:
        asyncio.run(run_interview(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nInterview session terminated by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
