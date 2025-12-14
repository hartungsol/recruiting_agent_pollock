"""Voice session loop (glue layer).

This module orchestrates:
mic -> STT -> orchestrator -> TTS -> playback

It intentionally does NOT re-implement interview logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from recruiting_agent_pollock.orchestrator.interview_orchestrator import InterviewOrchestrator
from recruiting_agent_pollock.orchestrator.schemas import CandidateProfile, InterviewConfig
from recruiting_agent_pollock.voice.stt import STTProvider, TranscriptionResult
from recruiting_agent_pollock.voice.tts import TTSProvider
from recruiting_agent_pollock.voice.speakable import to_speakable_question

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceSessionConfig:
    artifacts_dir: str = "data/interviews"
    push_to_talk: bool = True
    enable_human_interviewer_turns: bool = False
    min_transcript_chars: int = 2
    # Confidence heuristics (faster-whisper):
    min_avg_logprob: float = -1.2
    max_no_speech_prob: float = 0.9
    allow_barge_in: bool = False

    # TTS gating
    tts_enabled: bool = True
    tts_max_chars: int = 300
    tts_max_sentences: int = 2
    tts_speak_reasoning: bool = False  # must remain false; reasoning is never spoken
    tts_warmup: bool = True


class AudioIOProtocol(Protocol):
    async def start_recording(self) -> None: ...

    async def stop_recording(self) -> Any: ...

    def write_wav(self, wav_path: str | Path, audio: Any) -> Path: ...

    async def play_wav(self, wav_path: str | Path, *, allow_barge_in: bool = False) -> bool: ...


class VoiceSession:
    def __init__(
        self,
        *,
        orchestrator: InterviewOrchestrator,
        audio: AudioIOProtocol,
        stt: STTProvider,
        tts: TTSProvider,
        config: VoiceSessionConfig | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._audio = audio
        self._stt = stt
        self._tts = tts
        self._config = config or VoiceSessionConfig()

        self._session_dir: Path | None = None
        self._tts_disabled: bool = False
        self._turn_log_path: Path | None = None
        self._tts_cache_dir: Path | None = None

    @property
    def config(self) -> VoiceSessionConfig:
        return self._config

    async def start(self, *, interview_config: InterviewConfig, candidate: CandidateProfile) -> str:
        intro = await self._orchestrator.start_interview(interview_config, candidate)

        self._session_dir = self._make_session_dir()
        self._turn_log_path = self._session_dir / "turns.jsonl"
        self._tts_cache_dir = (self._session_dir / "tts_cache")
        self._tts_cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta(interview_config=interview_config, candidate=candidate)

        # Optional warm-up to reduce first-turn latency.
        if self._config.tts_enabled and self._config.tts_warmup:
            await self._tts_warmup()

        await self._speak(intro, base_name="intro")
        return intro

    async def step_from_transcript(self, transcript: str) -> str | None:
        """Run one orchestrator turn from already-transcribed candidate text."""
        text = (transcript or "").strip()
        if not text:
            return None

        if self._is_termination(text):
            result = await self._orchestrator.end_interview()
            closing = "Okay, ending the interview now. Thank you."
            await self._speak(closing, base_name="end")
            self._write_result(result)
            return None

        print("[Voice] Thinking...", flush=True)
        response = await self._orchestrator.process_candidate_input(text)
        await self._speak(response, base_name="reply")

        # If the orchestrator ended the interview internally, persist results.
        if not self._orchestrator.is_active:
            try:
                result = await self._orchestrator.end_interview()
                self._write_result(result)
            except Exception:
                pass
        return response

    async def run(self, *, interview_config: InterviewConfig, candidate: CandidateProfile) -> None:
        await self.start(interview_config=interview_config, candidate=candidate)

        try:
            while self._orchestrator.is_active:
                role, transcript = await self._capture_and_transcribe_any()
                if transcript is None:
                    continue
                if role == "candidate":
                    await self.step_from_transcript(transcript)
                else:
                    await self.step_interviewer_from_transcript(transcript)
        except KeyboardInterrupt:
            try:
                await self._audio.stop_recording()
            except Exception:
                pass
            if self._orchestrator.is_active:
                try:
                    result = await self._orchestrator.end_interview()
                    self._write_result(result)
                except Exception:
                    pass
            return

    async def step_interviewer_from_transcript(self, transcript: str) -> None:
        """Record a human interviewer turn from already-transcribed text."""
        text = (transcript or "").strip()
        if not text:
            return

        if self._is_termination(text):
            result = await self._orchestrator.end_interview()
            closing = "Okay, ending the interview now. Thank you."
            await self._speak(closing, base_name="end")
            self._write_result(result)
            return

        if not hasattr(self._orchestrator, "record_interviewer_input"):
            raise RuntimeError("Orchestrator does not support record_interviewer_input")

        self._orchestrator.record_interviewer_input(text)
        self._log_turn(role="interviewer", text=text, wavs=[])
        print(f"\n[Interviewer] {text}\n")

    async def _capture_and_transcribe_any(self) -> tuple[str, str | None]:
        role = "candidate"
        if self._config.enable_human_interviewer_turns:
            choice = await asyncio.to_thread(
                input,
                "\n[Voice] Enter=candidate, i=interviewer (human) ... ",
            )
            if (choice or "").strip().lower() == "i":
                role = "interviewer"

        transcript = await self._capture_and_transcribe(role=role)
        return role, transcript

    async def _capture_and_transcribe(self, *, role: str = "candidate") -> str | None:
        # Push-to-talk implementation using Enter prompts.
        if self._config.push_to_talk:
            print("\n[Voice] Waiting for Enter to start recording...", flush=True)
            await asyncio.to_thread(input, "\n[Voice] Press Enter to start recording... ")
            await self._audio.start_recording()
            await asyncio.to_thread(input, "[Voice] Recording... press Enter to stop. ")
            audio = await self._audio.stop_recording()
        else:
            # Minimal auto mode: record a fixed duration is intentionally not implemented.
            print("\n[Voice] Waiting for Enter to record...", flush=True)
            await asyncio.to_thread(input, "\n[Voice] (Auto mode not enabled) Press Enter to record... ")
            await self._audio.start_recording()
            await asyncio.sleep(4.0)
            audio = await self._audio.stop_recording()

        if getattr(audio, "size", 0) == 0:
            await self._speak("I didn't catch anything. Please try again.", base_name="empty")
            return None

        session_dir = self._session_dir or self._make_session_dir()
        turn_idx = len(self._orchestrator.current_state.turns) if self._orchestrator.current_state else 0
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "candidate" if role == "candidate" else "interviewer"
        wav_path = session_dir / f"turn_{turn_idx:03d}_{ts}_{suffix}.wav"
        self._audio.write_wav(wav_path, audio)

        print("[Voice] Transcribing (first run may download model)...", flush=True)
        result: TranscriptionResult = await self._stt.transcribe_file(wav_path)
        text = (result.text or "").strip()

        if len(text) < self._config.min_transcript_chars:
            await self._speak("I may have misheard. Could you repeat that?", base_name="lowtext")
            return None

        if result.no_speech_prob is not None and result.no_speech_prob >= self._config.max_no_speech_prob:
            await self._speak("I didn't hear clear speech. Please repeat.", base_name="nospeech")
            return None

        if result.avg_logprob is not None and result.avg_logprob <= self._config.min_avg_logprob:
            await self._speak("I may have misheard. Could you repeat that?", base_name="lowconf")
            return None

        # Persist transcript alongside the audio.
        transcript_path = session_dir / f"turn_{turn_idx:03d}_{ts}_{suffix}.txt"
        transcript_path.write_text(text + "\n", encoding="utf-8")
        self._log_turn(role=role, text=text, wavs=[str(wav_path.name), str(transcript_path.name)])

        label = "Candidate" if role == "candidate" else "Interviewer"
        print(f"\n[{label}] {text}\n")
        return text

    async def _speak(self, text: str, *, base_name: str) -> None:
        # Always record the visible text in the transcript.
        if not self._config.tts_enabled or self._tts_disabled:
            print(f"\n[Interviewer] {text}\n")
            self._log_turn(role="interviewer", text=text, wavs=[])
            if not self._config.tts_enabled:
                logger.info("[VOICE][TTS] skipped reason=tts_disabled")
            return

        # Enforce strict speakable filtering (never speak reasoning/JSON/long output).
        speakable, dbg = to_speakable_question(
            text,
            max_chars=self._config.tts_max_chars,
            max_sentences=self._config.tts_max_sentences,
            # Even if someone sets this true, ignore it for safety.
            speak_reasoning=False,
        )

        if speakable is None:
            reason = dbg.get("skip_reason") or "no_speakable_text"
            logger.info(f"[VOICE][TTS] skipped reason={reason} debug={dbg}")
            print(f"\n[Interviewer] {text}\n")
            self._log_turn(role="interviewer", text=text, wavs=[])
            return

        session_dir = self._session_dir or self._make_session_dir()
        cache_dir = self._tts_cache_dir or (session_dir / "tts_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache by (model-ish settings + speakable text), so repeated prompts are instant.
        cache_key = hashlib.sha1(
            (speakable + "\n" + str(getattr(getattr(self._tts, "config", None), "model_path", ""))).encode("utf-8")
        ).hexdigest()[:16]
        cached = sorted(cache_dir.glob(f"{cache_key}_*.wav"))
        if cached:
            wavs = cached
        else:
            t0 = time.perf_counter()
            try:
                wavs = await self._tts.synthesize_to_wavs(
                    speakable,
                    out_dir=cache_dir,
                    base_name=cache_key,
                )
            except RuntimeError as e:
                # Allow voice mode to run even when Piper (or another TTS backend) isn't installed.
                logger.warning(f"TTS unavailable; falling back to text output: {e}")
                self._tts_disabled = True
                print(f"\n[Interviewer] {text}\n")
                self._log_turn(role="interviewer", text=text, wavs=[])
                return
            dur = time.perf_counter() - t0
            excerpt = speakable[:80].replace("\n", " ")
            logger.info(
                f"[VOICE][TTS] speak len={len(speakable)} sent={dbg.get('sentences')} dur={dur:.2f}s text=\"{excerpt}\" debug={dbg}"
            )

        if not wavs:
            return

        wav_names: list[str] = []
        for wav in wavs:
            name = getattr(wav, "name", None)
            wav_names.append(str(name) if name is not None else Path(str(wav)).name)

        # Log the original (full) public text but attach the spoken WAVs.
        self._log_turn(role="interviewer", text=text, wavs=wav_names)
        for wav in wavs:
            interrupted = await self._audio.play_wav(str(wav), allow_barge_in=self._config.allow_barge_in)
            if interrupted:
                logger.info(
                    "[VOICE][AUDIO] playback interrupted (barge_in=%s) wav=%s",
                    self._config.allow_barge_in,
                    str(wav),
                )
                break

    async def _tts_warmup(self) -> None:
        # Warm up Piper once with a short phrase, cached and not played.
        if self._tts_disabled:
            return

        session_dir = self._session_dir or self._make_session_dir()
        cache_dir = self._tts_cache_dir or (session_dir / "tts_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        phrase = "Voice system ready."
        speakable, dbg = to_speakable_question(
            phrase,
            max_chars=self._config.tts_max_chars,
            max_sentences=self._config.tts_max_sentences,
            speak_reasoning=False,
        )
        if not speakable:
            logger.info(f"[VOICE][TTS] warmup skipped debug={dbg}")
            return

        cache_key = hashlib.sha1(("warmup\n" + speakable).encode("utf-8")).hexdigest()[:16]
        if list(cache_dir.glob(f"{cache_key}_*.wav")):
            return

        t0 = time.perf_counter()
        try:
            _ = await self._tts.synthesize_to_wavs(speakable, out_dir=cache_dir, base_name=cache_key)
        except Exception as e:
            logger.info(f"[VOICE][TTS] warmup failed: {e}")
            return
        dur = time.perf_counter() - t0
        logger.info(f"[VOICE][TTS] warmup ok dur={dur:.2f}s")

    def _make_session_dir(self) -> Path:
        base = Path(self._config.artifacts_dir)
        interview_id = "unknown"
        if self._orchestrator.current_state:
            interview_id = str(self._orchestrator.current_state.interview_id)

        d = base / interview_id
        d.mkdir(parents=True, exist_ok=True)
        self._session_dir = d
        return d

    def _write_meta(self, *, interview_config: InterviewConfig, candidate: CandidateProfile) -> None:
        if not self._session_dir:
            return
        meta = {
            "interview_id": str(self._orchestrator.current_state.interview_id) if self._orchestrator.current_state else None,
            "started_at": datetime.now().isoformat(),
            "candidate": {"name": candidate.name, "email": candidate.email, "experience_level": candidate.experience_level},
            "job": {"job_id": interview_config.job_id, "job_title": interview_config.job_title, "company_name": interview_config.company_name},
        }
        (self._session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _write_result(self, result: Any) -> None:
        if not self._session_dir:
            return
        try:
            data = result.model_dump()  # pydantic
        except Exception:
            try:
                data = result.__dict__
            except Exception:
                data = {"result": str(result)}
        (self._session_dir / "result.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _log_turn(self, *, role: str, text: str, wavs: list[str] | None) -> None:
        if not self._turn_log_path:
            return
        rec = {
            "ts": datetime.now().isoformat(),
            "role": role,
            "text": text,
            "artifacts": wavs or [],
        }
        with self._turn_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _is_termination(text: str) -> bool:
        t = text.lower().strip()
        return any(
            phrase in t
            for phrase in (
                "stop",
                "end interview",
                "end the interview",
                "quit",
                "exit",
            )
        )
