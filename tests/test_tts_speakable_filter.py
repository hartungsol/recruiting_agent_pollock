import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from recruiting_agent_pollock.voice.speakable import to_speakable_question
from recruiting_agent_pollock.voice.voice_session import VoiceSession, VoiceSessionConfig
from recruiting_agent_pollock.orchestrator.schemas import CandidateProfile, InterviewConfig
from recruiting_agent_pollock.voice.stt import STTProvider
from recruiting_agent_pollock.voice.tts import TTSProvider


class FakeAudio:
    async def start_recording(self) -> None:
        raise AssertionError("start_recording should not be called")

    async def stop_recording(self):
        raise AssertionError("stop_recording should not be called")

    def write_wav(self, wav_path, audio):
        raise AssertionError("write_wav should not be called")

    async def play_wav(self, wav_path: str | Path, *, allow_barge_in: bool = False) -> bool:
        return False


class FakeSTT(STTProvider):
    async def transcribe_file(self, wav_path):
        raise AssertionError("transcribe_file should not be called")


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize_to_wavs(self, text: str, out_dir, base_name: str):
        self.calls.append(text)
        return [f"{out_dir}/{base_name}_00.wav"]


@dataclass
class _State:
    interview_id: str = "test-interview"
    turns: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.turns is None:
            self.turns = []


class StubOrchestrator:
    def __init__(self, *, intro: str, reply: str) -> None:
        self._intro = intro
        self._reply = reply
        self.is_active = True
        self.current_state = _State()

    async def start_interview(self, interview_config, candidate):
        self.current_state.turns.append({"role": "interviewer", "content": self._intro})
        return self._intro

    async def process_candidate_input(self, candidate_text: str):
        self.current_state.turns.append({"role": "candidate", "content": candidate_text})
        return self._reply

    async def end_interview(self):
        self.is_active = False
        return "Goodbye."


def test_to_speakable_strips_reasoning_prefers_answer_and_first_question():
    text = """<reasoning>LOTS OF INTERNAL STUFF</reasoning>
<answer>Thanks for that. Next question: What is your experience with Python? Also, what about SQL?</answer>"""

    speak, dbg = to_speakable_question(text, max_chars=300, max_sentences=2, speak_reasoning=False)
    assert speak == "What is your experience with Python?"
    assert dbg["stripped_reasoning"] is True
    assert dbg["used_answer_tag"] is True
    assert dbg["selected_question"] is True


def test_to_speakable_skips_json_and_code_fences():
    speak, dbg = to_speakable_question('{"a": 1, "b": 2}')
    assert speak is None
    assert dbg["skip_reason"] == "contained_json"

    speak2, dbg2 = to_speakable_question("```json\n{\"a\": 1}\n```")
    assert speak2 is None
    assert dbg2["skip_reason"] == "contained_code_fence"


def test_voice_session_speaks_only_short_question(tmp_path):
    intro = "<reasoning>hidden</reasoning><answer>Welcome. First question: Tell me about yourself?</answer>"
    reply = "<reasoning>scorecard blah</reasoning>Here are the results...\n\nWhat is your proudest project? Why?"

    tts = FakeTTS()
    session = VoiceSession(
        orchestrator=cast(Any, StubOrchestrator(intro=intro, reply=reply)),
        audio=FakeAudio(),
        stt=FakeSTT(),
        tts=tts,
        config=VoiceSessionConfig(artifacts_dir=str(tmp_path), tts_warmup=False),
    )

    async def _run():
        await session.start(
            interview_config=InterviewConfig(job_id="j", job_title="T"),
            candidate=CandidateProfile(name="N", email="e@example.com"),
        )
        await session.step_from_transcript("hello")

    asyncio.run(_run())

    # Intro speaks only the question
    assert tts.calls[0] == "Tell me about yourself?"
    # Reply speaks only the first question
    assert tts.calls[1] == "What is your proudest project?"


def test_voice_session_skips_tts_for_json(tmp_path):
    tts = FakeTTS()
    session = VoiceSession(
        orchestrator=cast(Any, StubOrchestrator(intro="Welcome.", reply='{"score": 0.5, "red_flags": []}')),
        audio=FakeAudio(),
        stt=FakeSTT(),
        tts=tts,
        config=VoiceSessionConfig(artifacts_dir=str(tmp_path), tts_warmup=False),
    )

    async def _run():
        await session.start(
            interview_config=InterviewConfig(job_id="j", job_title="T"),
            candidate=CandidateProfile(name="N", email="e@example.com"),
        )
        await session.step_from_transcript("hello")

    asyncio.run(_run())

    # Only the intro should have been spoken; the JSON reply should be skipped.
    assert len(tts.calls) == 1
