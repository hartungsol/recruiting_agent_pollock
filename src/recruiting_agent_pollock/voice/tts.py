"""Text-to-speech (offline).

Default implementation uses `piper` via subprocess if available.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSConfig:
    piper_bin: str = "piper"
    model_path: str | None = None  # path to *.onnx
    speaker_id: int | None = None
    max_chars_per_chunk: int = 350
    timeout_s: float = 60.0


class TTSProvider:
    async def synthesize_to_wavs(self, text: str, out_dir: str | Path, base_name: str) -> list[Path]:
        raise NotImplementedError


class PiperTTS(TTSProvider):
    def __init__(self, config: TTSConfig | None = None) -> None:
        self._config = config or TTSConfig()
        self._validated_piper_path: str | None = None

    @property
    def config(self) -> TTSConfig:
        return self._config

    def is_available(self) -> tuple[bool, str]:
        try:
            _ = self._require_piper()
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def _looks_like_piper_tts(self, piper_path: str) -> bool:
        try:
            r = subprocess.run(
                [piper_path, "--help"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
        except Exception:
            return False

        out = (r.stdout or "").lower()
        # Piper TTS CLI typically supports these flags.
        return ("--model" in out or "--output_file" in out) and "application options" not in out

    def _require_piper(self) -> str:
        if self._validated_piper_path:
            return self._validated_piper_path

        p = shutil.which(self._config.piper_bin)
        if not p:  # pragma: no cover
            raise RuntimeError(
                "piper CLI not found. Install piper (binary) and ensure it's on PATH, "
                "or set TTSConfig(piper_bin=...)."
            )
        if not self._looks_like_piper_tts(p):  # pragma: no cover
            raise RuntimeError(
                "Found a `piper` binary, but it does not look like the Piper TTS CLI (common on Linux: /usr/bin/piper is a GTK app). "
                "Install Piper TTS and set POLL0CK_PIPER_BIN to that binary path (e.g., ~/piper/piper), then retry."
            )
        if not self._config.model_path:  # pragma: no cover
            raise RuntimeError(
                "Piper model path not configured. Set TTSConfig(model_path='/path/to/voice.onnx')."
            )

        self._validated_piper_path = p
        return p

    def _chunk_text(self, text: str) -> list[str]:
        t = (text or "").strip()
        if not t:
            return []

        # Split on sentence-ish boundaries, then re-pack into chunks.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
        chunks: list[str] = []
        current = ""
        for p in parts:
            if not current:
                current = p
                continue
            if len(current) + 1 + len(p) <= self._config.max_chars_per_chunk:
                current = current + " " + p
            else:
                chunks.append(current)
                current = p
        if current:
            chunks.append(current)

        # Fallback if there were no sentence boundaries.
        if not chunks and t:
            for i in range(0, len(t), self._config.max_chars_per_chunk):
                chunks.append(t[i : i + self._config.max_chars_per_chunk])

        return chunks

    async def synthesize_to_wavs(self, text: str, out_dir: str | Path, base_name: str) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        piper_bin = self._require_piper()
        chunks = self._chunk_text(text)
        if not chunks:
            return []

        wavs: list[Path] = []

        async def _run_one(chunk: str, wav_path: Path) -> None:
            cmd = [piper_bin, "--model", str(self._config.model_path), "--output_file", str(wav_path)]
            if self._config.speaker_id is not None:
                cmd += ["--speaker", str(self._config.speaker_id)]

            def _call() -> None:
                try:
                    subprocess.run(
                        cmd,
                        input=chunk,
                        text=True,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=self._config.timeout_s,
                    )
                except subprocess.TimeoutExpired as e:  # pragma: no cover
                    raise RuntimeError(
                        f"piper timed out after {self._config.timeout_s:.1f}s. "
                        f"model={self._config.model_path!s}. "
                        "Consider reducing max_chars_per_chunk or increasing timeout_s."
                    ) from e
                except subprocess.CalledProcessError as e:  # pragma: no cover
                    stderr = (e.stderr or "").strip()
                    raise RuntimeError(
                        f"piper failed (exit={e.returncode}). "
                        f"model={self._config.model_path!s}. "
                        f"stderr={stderr or '<empty>'}"
                    ) from e

            await asyncio.to_thread(_call)

        for idx, chunk in enumerate(chunks):
            wav_path = out_dir / f"{base_name}_{idx:02d}.wav"
            await _run_one(chunk, wav_path)
            wavs.append(wav_path)

        return wavs
