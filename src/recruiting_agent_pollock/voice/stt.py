"""Speech-to-text (offline).

Default implementation uses `faster-whisper` if installed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STTConfig:
    model_size: str = "small"
    # Default to CPU to avoid hard crashes when CUDA/cuDNN aren't present.
    device: str = "cpu"  # cpu|cuda|auto
    compute_type: str | None = None  # e.g. int8, float16
    language: str | None = None
    vad_filter: bool = True


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class STTProvider:
    async def transcribe_file(self, wav_path: str | Path) -> TranscriptionResult:
        raise NotImplementedError


class WhisperSTT(STTProvider):
    """faster-whisper wrapper."""

    def __init__(self, config: STTConfig | None = None) -> None:
        self._config = config or STTConfig()
        self._model = None

    @property
    def config(self) -> STTConfig:
        return self._config

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is required for STT. Install with: pip install -e '.[voice]'"
            ) from e

        device = self._config.device
        if device == "auto":
            # Be conservative: prefer CPU unless user explicitly requests CUDA.
            device = "cpu"

        kwargs = {}
        if self._config.compute_type:
            kwargs["compute_type"] = self._config.compute_type

        self._model = WhisperModel(self._config.model_size, device=device, **kwargs)
        return self._model

    async def transcribe_file(self, wav_path: str | Path) -> TranscriptionResult:
        wav_path = Path(wav_path)

        def _run() -> TranscriptionResult:
            model = self._load_model()
            segments, info = model.transcribe(
                str(wav_path),
                language=self._config.language,
                vad_filter=self._config.vad_filter,
            )
            text_parts: list[str] = []
            for s in segments:
                if s.text:
                    text_parts.append(s.text.strip())
            text = " ".join(t for t in text_parts if t).strip()
            avg_logprob = getattr(info, "avg_logprob", None)
            no_speech_prob = getattr(info, "no_speech_prob", None)
            return TranscriptionResult(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)

        return await asyncio.to_thread(_run)
