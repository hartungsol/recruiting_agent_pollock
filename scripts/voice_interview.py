#!/usr/bin/env python

import argparse
import asyncio
import os
import sys

from recruiting_agent_pollock.config import get_settings
from recruiting_agent_pollock.models.llm_client import LLMClient
from recruiting_agent_pollock.orchestrator.interview_orchestrator import InterviewOrchestrator
from recruiting_agent_pollock.orchestrator.job_ingestion import JobIngestionService
from recruiting_agent_pollock.orchestrator.schemas import CandidateProfile, InterviewConfig
from recruiting_agent_pollock.voice.audio_io import AudioIO, AudioIOConfig
from recruiting_agent_pollock.voice.stt import STTConfig, WhisperSTT
from recruiting_agent_pollock.voice.tts import PiperTTS, TTSConfig
from recruiting_agent_pollock.voice.voice_session import VoiceSession, VoiceSessionConfig


async def _load_job_config(args, llm_client: LLMClient) -> InterviewConfig:
    ingestion = JobIngestionService(llm_client=llm_client)

    if args.job_file:
        job = await ingestion.ingest_from_file(args.job_file)
        return job.to_interview_config()

    if args.job_text:
        job = await ingestion.ingest_from_text(args.job_text)
        return job.to_interview_config()

    if args.stdin_job:
        if sys.stdin.isatty():
            raise RuntimeError(
                "--stdin-job was set, but stdin is a TTY (nothing is being piped). "
                "Pipe the job description into stdin or use --job-text / --job-file instead."
            )

        raw = sys.stdin.read()
        if raw.strip():
            job = await ingestion.ingest_from_text(raw)
            return job.to_interview_config()

    # Fallback: minimal config
    return InterviewConfig(job_id="cli-job", job_title=args.job_title)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Recruiting Agent Pollock in voice mode")
    p.add_argument("--candidate-name", required=True)
    p.add_argument("--candidate-email", default="")
    p.add_argument("--experience-level", default="unknown", choices=["entry", "mid", "senior", "unknown"])

    g = p.add_mutually_exclusive_group()
    g.add_argument("--job-file", help="Path to a job description file")
    g.add_argument("--job-text", help="Job description text")
    g.add_argument("--stdin-job", action="store_true", help="Read job description from stdin")

    p.add_argument("--job-title", default="demo-job", help="Used if no job description provided")

    p.add_argument("--artifacts-dir", default="data/interviews", help="Where to store interview artifacts")
    p.add_argument(
        "--human-interviewer",
        action="store_true",
        help="Allow recording/transcribing human interviewer turns (choose 'i' at the prompt)",
    )

    p.add_argument("--sample-rate", type=int, default=16000)

    # TTS gating
    p.add_argument(
        "--tts-enabled",
        default=os.getenv("VOICE_TTS_ENABLED", "true"),
        help="Enable TTS (default: VOICE_TTS_ENABLED or true)",
    )
    p.add_argument(
        "--tts-max-chars",
        type=int,
        default=int(os.getenv("VOICE_TTS_MAX_CHARS", "300") or "300"),
        help="Max chars to send to TTS (default: VOICE_TTS_MAX_CHARS or 300)",
    )
    p.add_argument(
        "--tts-max-sentences",
        type=int,
        default=int(os.getenv("VOICE_TTS_MAX_SENTENCES", "2") or "2"),
        help="Max sentences to send to TTS (default: VOICE_TTS_MAX_SENTENCES or 2)",
    )
    p.add_argument(
        "--tts-warmup",
        default=os.getenv("VOICE_TTS_WARMUP", "true"),
        help="Warm up TTS at startup (default: VOICE_TTS_WARMUP or true)",
    )

    # Playback
    p.add_argument(
        "--allow-barge-in",
        default=os.getenv("VOICE_ALLOW_BARGE_IN", "false"),
        help="Stop playback when mic noise is detected (default: VOICE_ALLOW_BARGE_IN or false)",
    )

    # STT
    p.add_argument(
        "--stt-model",
        default=os.getenv("POLL0CK_STT_MODEL", "small"),
        help="faster-whisper model size (default: POLL0CK_STT_MODEL or 'small')",
    )
    p.add_argument(
        "--stt-device",
        default=os.getenv("POLL0CK_STT_DEVICE", "cpu"),
        choices=["cpu", "cuda", "auto"],
        help="STT device (default: POLL0CK_STT_DEVICE or 'cpu')",
    )

    # TTS
    p.add_argument(
        "--piper-bin",
        default=os.getenv("POLL0CK_PIPER_BIN", "piper"),
        help="Path/name of Piper TTS binary (default: POLL0CK_PIPER_BIN or 'piper')",
    )
    p.add_argument(
        "--piper-model",
        default=os.getenv("POLL0CK_PIPER_MODEL", None),
        help="Path to Piper .onnx model (default: POLL0CK_PIPER_MODEL)",
    )
    p.add_argument(
        "--piper-timeout",
        type=float,
        default=float(os.getenv("POLL0CK_PIPER_TIMEOUT_S", "60") or "60"),
        help="Timeout (seconds) per Piper synthesis chunk (default: POLL0CK_PIPER_TIMEOUT_S or 60)",
    )

    return p


async def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    def _flag(v: str) -> bool:
        return (v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    settings = get_settings()
    llm_client = LLMClient(model=settings.llm_model_name, timeout=settings.llm_timeout)
    orchestrator = InterviewOrchestrator(llm_client=llm_client)

    candidate = CandidateProfile(
        name=args.candidate_name,
        email=args.candidate_email,
        experience_level=args.experience_level,
    )

    config = await _load_job_config(args, llm_client)

    audio = AudioIO(AudioIOConfig(sample_rate=args.sample_rate))
    stt = WhisperSTT(STTConfig(model_size=args.stt_model, device=args.stt_device))
    # Prefer an explicit per-chunk timeout to avoid indefinite hangs.
    tts = PiperTTS(TTSConfig(piper_bin=args.piper_bin, model_path=args.piper_model, timeout_s=args.piper_timeout))

    session = VoiceSession(
        orchestrator=orchestrator,
        audio=audio,
        stt=stt,
        tts=tts,
        config=VoiceSessionConfig(
            artifacts_dir=args.artifacts_dir,
            enable_human_interviewer_turns=bool(args.human_interviewer),
            tts_enabled=_flag(args.tts_enabled),
            tts_max_chars=int(args.tts_max_chars),
            tts_max_sentences=int(args.tts_max_sentences),
            tts_speak_reasoning=False,
            tts_warmup=_flag(args.tts_warmup),
            allow_barge_in=_flag(args.allow_barge_in),
        ),
    )

    await session.run(interview_config=config, candidate=candidate)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
