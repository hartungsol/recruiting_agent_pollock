import pytest


def test_voice_runner_env_defaults_are_used(monkeypatch):
    # This test is intentionally lightweight: it verifies our CLI defaults
    # are wired to environment variables, matching `--mode voice` behavior.
    monkeypatch.setenv("POLL0CK_PIPER_BIN", "/tmp/piper")
    monkeypatch.setenv("POLL0CK_PIPER_MODEL", "/tmp/voice.onnx")
    monkeypatch.setenv("POLL0CK_STT_MODEL", "small")
    monkeypatch.setenv("POLL0CK_STT_DEVICE", "cpu")

    from scripts.voice_interview import build_parser

    args = build_parser().parse_args(["--candidate-name", "Ted", "--stdin-job"])
    assert args.piper_bin == "/tmp/piper"
    assert args.piper_model == "/tmp/voice.onnx"
    assert args.stt_model == "small"
    assert args.stt_device == "cpu"


@pytest.mark.asyncio
async def test_voice_runner_stdin_job_tty_guard(monkeypatch):
    from scripts import voice_interview

    class _Args:
        job_file = None
        job_text = None
        stdin_job = True
        job_title = "demo"

    class _FakeStdin:
        def isatty(self):
            return True

        def read(self):
            raise AssertionError("stdin.read() should not be called when stdin is a TTY")

    monkeypatch.setattr(voice_interview.sys, "stdin", _FakeStdin())

    with pytest.raises(RuntimeError, match="stdin is a TTY"):
        # Pass a dummy llm_client; it won't be used due to fail-fast guard.
        await voice_interview._load_job_config(_Args(), llm_client=None)
