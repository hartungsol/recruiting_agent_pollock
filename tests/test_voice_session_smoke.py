import asyncio

from recruiting_agent_pollock.voice.voice_session import VoiceSession, VoiceSessionConfig
from recruiting_agent_pollock.orchestrator.schemas import CandidateProfile, InterviewConfig


class FakeAudio:
    def __init__(self) -> None:
        self.played: list[str] = []
        self._stop_playback_called = False

    async def start_recording(self) -> None:
        raise AssertionError("start_recording should not be called")

    async def stop_recording(self):
        raise AssertionError("stop_recording should not be called")

    def write_wav(self, path, audio) -> None:
        raise AssertionError("write_wav should not be called")

    async def play_wav(self, wav_path: str, *, allow_barge_in: bool = False) -> bool:
        self.played.append(str(wav_path))
        return False

    def stop_playback(self) -> None:
        self._stop_playback_called = True


class FakeSTT:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    async def transcribe_file(self, wav_path):
        if not self._texts:
            raise AssertionError("No more fake STT outputs")
        text = self._texts.pop(0)
        return type(
            "TranscriptionResult",
            (),
            {
                "text": text,
                "language": "en",
                "segments": [],
                "duration_s": None,
                "avg_logprob": None,
                "no_speech_prob": None,
            },
        )()


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._counter = 0

    async def synthesize_to_wavs(self, text: str, out_dir, base_name: str):
        self.spoken.append(text)
        self._counter += 1
        # Return a fake path; AudioIO is also fake.
        return [f"{out_dir}/{base_name}_{self._counter}.wav"]


class StubOrchestrator:
    def __init__(self) -> None:
        self.is_active = True
        self.current_state = type(
            "State",
            (),
            {"interview_id": "test-interview", "turns": []},
        )()

    async def start_interview(self, interview_config, candidate):
        # Simulate the orchestrator adding the interviewer intro turn.
        self.current_state.turns.append({"role": "interviewer", "content": "Welcome."})
        return "Welcome."

    async def process_candidate_input(self, candidate_text: str):
        self.current_state.turns.append({"role": "candidate", "content": candidate_text})
        return "Thanks."

    async def end_interview(self):
        self.is_active = False
        return "Goodbye."

    def record_interviewer_input(self, input_text: str, *, source: str = "human_interviewer") -> None:
        self.current_state.turns.append({"role": "interviewer", "content": input_text, "source": source})


def test_voice_session_stop_phrase_ends_cleanly(tmp_path):
    audio = FakeAudio()
    stt = FakeSTT(["stop"])  # not used
    tts = FakeTTS()
    orchestrator = StubOrchestrator()

    session = VoiceSession(
        orchestrator=orchestrator,
        audio=audio,
        stt=stt,
        tts=tts,
        config=VoiceSessionConfig(artifacts_dir=str(tmp_path)),
    )

    async def _run():
        await session.start(
            interview_config=InterviewConfig(job_id="j", job_title="T"),
            candidate=CandidateProfile(name="N", email="e@example.com"),
        )
        await session.step_from_transcript("stop")

    asyncio.run(_run())
    assert tts.spoken, "Expected at least one TTS utterance"
    assert orchestrator.is_active is False


def test_voice_session_records_human_interviewer_turn(tmp_path):
    audio = FakeAudio()
    stt = FakeSTT(["hello"])  # not used
    tts = FakeTTS()
    orchestrator = StubOrchestrator()

    session = VoiceSession(
        orchestrator=orchestrator,
        audio=audio,
        stt=stt,
        tts=tts,
        config=VoiceSessionConfig(artifacts_dir=str(tmp_path)),
    )

    async def _run():
        await session.start(
            interview_config=InterviewConfig(job_id="j", job_title="T"),
            candidate=CandidateProfile(name="N", email="e@example.com"),
        )
        await session.step_interviewer_from_transcript("Can you walk me through your last role?")

    asyncio.run(_run())
    assert any(
        t.get("role") == "interviewer" and "last role" in t.get("content", "")
        for t in orchestrator.current_state.turns
    )
