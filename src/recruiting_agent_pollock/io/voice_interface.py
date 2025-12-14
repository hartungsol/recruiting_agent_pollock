"""Voice-based interview interface.

This file stays intentionally thin: it delegates the actual voice loop to the
modular subsystem in `recruiting_agent_pollock.voice.*`.

The orchestrator remains the single authority for interview flow and memory.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from recruiting_agent_pollock.io.text_interface import InterviewInterface
from recruiting_agent_pollock.models.llm_client import LLMClient
from recruiting_agent_pollock.orchestrator.interview_orchestrator import InterviewOrchestrator
from recruiting_agent_pollock.orchestrator.job_ingestion import JobIngestionService
from recruiting_agent_pollock.orchestrator.schemas import CandidateProfile, InterviewConfig

if TYPE_CHECKING:
    from recruiting_agent_pollock.voice.audio_io import AudioIO
    from recruiting_agent_pollock.voice.stt import WhisperSTT
    from recruiting_agent_pollock.voice.tts import PiperTTS
    from recruiting_agent_pollock.voice.voice_session import VoiceSession, VoiceSessionConfig


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


class VoiceInterface(InterviewInterface):
    def __init__(
        self,
        orchestrator: InterviewOrchestrator,
        *,
        llm_client: LLMClient | None = None,
        audio: "AudioIO | None" = None,
        stt: "WhisperSTT | None" = None,
        tts: "PiperTTS | None" = None,
        session_config: "VoiceSessionConfig | None" = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._llm_client = llm_client
        self._job_ingestion = JobIngestionService(llm_client=llm_client)

        try:
            from recruiting_agent_pollock.voice.audio_io import AudioIO, AudioIOConfig
            from recruiting_agent_pollock.voice.stt import STTConfig, WhisperSTT
            from recruiting_agent_pollock.voice.tts import PiperTTS, TTSConfig
            from recruiting_agent_pollock.voice.voice_session import VoiceSession, VoiceSessionConfig
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Voice mode dependencies are not installed. Install with: pip install -e '.[voice]'. "
                "Also install `piper` (CLI) and download a Piper .onnx voice model, and install `faster-whisper` "
                "if you want offline STT."
            ) from e

        # Defaults are offline/local components.
        self._audio = audio or AudioIO(AudioIOConfig())

        stt_cfg = STTConfig(
            model_size=os.getenv("POLL0CK_STT_MODEL", "small"),
            device=os.getenv("POLL0CK_STT_DEVICE", "cpu"),
        )
        self._stt = stt or WhisperSTT(stt_cfg)

        tts_cfg = TTSConfig(
            piper_bin=os.getenv("POLL0CK_PIPER_BIN", "piper"),
            model_path=os.getenv("POLL0CK_PIPER_MODEL", None),
            timeout_s=float(os.getenv("POLL0CK_PIPER_TIMEOUT_S", "60") or "60"),
        )
        self._tts = tts or PiperTTS(tts_cfg)

        self._session = VoiceSession(
            orchestrator=self._orchestrator,
            audio=self._audio,
            stt=self._stt,
            tts=self._tts,
            config=session_config
            or VoiceSessionConfig(
                tts_enabled=_env_flag("VOICE_TTS_ENABLED", True),
                tts_max_chars=int(os.getenv("VOICE_TTS_MAX_CHARS", "300") or "300"),
                tts_max_sentences=int(os.getenv("VOICE_TTS_MAX_SENTENCES", "2") or "2"),
                allow_barge_in=_env_flag("VOICE_ALLOW_BARGE_IN", False),
                # Safety: never speak reasoning, regardless of env var.
                tts_speak_reasoning=False,
                tts_warmup=_env_flag("VOICE_TTS_WARMUP", True),
            ),
        )

    async def run(self) -> None:
        print("\n" + "=" * 60)
        print("Recruiting Agent Pollock — Voice Mode")
        print("=" * 60 + "\n")

        # Explicitly report whether interviewer speech is configured.
        try:
            ok, reason = self._tts.is_available()
            if ok:
                piper_path = shutil.which(getattr(self._tts.config, "piper_bin", "piper"))
                model_path = getattr(self._tts.config, "model_path", None)
                print(f"Interviewer voice: ENABLED (piper='{piper_path}', model='{model_path}')")
            else:
                print(
                    "Interviewer voice: DISABLED. "
                    "Set POLL0CK_PIPER_MODEL=/path/to/voice.onnx and POLL0CK_PIPER_BIN=/path/to/piper-tts-binary. "
                    f"Reason: {reason}"
                )
        except Exception:
            # Stay non-fatal; VoiceSession will still fall back to printing.
            pass

        # Candidate/job intake stays text-based for now.
        print("Please enter candidate information (text):")
        name = await self._get_input("Candidate name: ")
        email = await self._get_input("Candidate email: ")
        candidate = CandidateProfile(name=name, email=email)

        config = await self._get_job_config()

        print("\n" + "-" * 60)
        print("Starting Voice Interview")
        print("-" * 60 + "\n")

        await self._session.run(interview_config=config, candidate=candidate)

    async def send_message(self, message: str) -> None:
        # For interface compatibility; voice mode speaks in VoiceSession.
        print(message)

    async def receive_input(self) -> str:
        # For interface compatibility; voice mode records in VoiceSession.
        return await self._get_input("You: ")

    async def _get_input(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            return "exit"

    async def _get_job_config(self) -> InterviewConfig:
        print("\nHow would you like to provide the job information?")
        print("  1. Enter job description text")
        print("  2. Load job description from file")
        print("  3. Quick setup (just job title)")

        choice = await self._get_input("\nChoice [1/2/3]: ")
        if choice == "1":
            return await self._get_job_from_text()
        if choice == "2":
            return await self._get_job_from_file()
        return await self._get_job_simple()

    async def _get_job_simple(self) -> InterviewConfig:
        job_title = await self._get_input("Job title: ")
        return InterviewConfig(job_id="demo-job", job_title=job_title)

    async def _get_job_from_text(self) -> InterviewConfig:
        print("\nPaste the job description below.\nEnter a blank line when done:\n")

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
            return job_description.to_interview_config()
        except Exception as e:
            print(f"Failed to parse job description: {e}")
            print("Falling back to simple setup.")
            return await self._get_job_simple()

    async def _get_job_from_file(self) -> InterviewConfig:
        file_path = (await self._get_input("File path: ")).strip()
        if not file_path:
            print("No file path provided. Using simple setup.")
            return await self._get_job_simple()

        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            print("File does not exist. Using simple setup.")
            return await self._get_job_simple()

        try:
            job_description = await self._job_ingestion.ingest_from_file(file_path)
            print(f"✓ Parsed: {job_description.title}")
            return job_description.to_interview_config()
        except Exception as e:
            print(f"Failed to parse job description: {e}")
            print("Falling back to simple setup.")
            return await self._get_job_simple()
