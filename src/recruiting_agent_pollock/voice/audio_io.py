"""Audio capture + playback (LLM-agnostic).

This module is intentionally "dumb hardware I/O": it knows nothing about the
interview, prompts, or LLMs.

It provides:
- push-to-talk microphone capture (start/stop)
- WAV saving/loading helpers
- speaker playback
- optional barge-in (stop playback if mic energy crosses a threshold)
"""

from __future__ import annotations

import asyncio
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioIOConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"  # sounddevice dtype and WAV sample width
    barge_in_rms_threshold: float = 0.02


class AudioIO:
    def __init__(self, config: AudioIOConfig | None = None) -> None:
        self._config = config or AudioIOConfig()
        self._recording_stream = None
        self._recording_frames: list[np.ndarray] = []

    @property
    def config(self) -> AudioIOConfig:
        return self._config

    def _require_sounddevice(self):
        try:
            import sounddevice as sd  # type: ignore

            return sd
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sounddevice is required for voice mode. Install Python deps with: pip install -e '.[voice]'. "
                "If you see 'PortAudio library not found', install PortAudio (Debian/Ubuntu: sudo apt-get install portaudio19-dev)."
            ) from e

    async def start_recording(self) -> None:
        """Start mic capture (push-to-talk)."""
        sd = self._require_sounddevice()
        self._recording_frames = []

        def callback(indata, frames, time, status):  # noqa: ANN001
            if status:
                logger.debug(f"Input status: {status}")
            self._recording_frames.append(indata.copy())

        self._recording_stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            callback=callback,
        )

        await asyncio.to_thread(self._recording_stream.start)

    async def stop_recording(self) -> np.ndarray:
        """Stop mic capture and return audio as int16 numpy array [samples, channels]."""
        if self._recording_stream is None:
            return np.zeros((0, self._config.channels), dtype=np.int16)

        stream = self._recording_stream
        self._recording_stream = None

        await asyncio.to_thread(stream.stop)
        await asyncio.to_thread(stream.close)

        if not self._recording_frames:
            return np.zeros((0, self._config.channels), dtype=np.int16)

        audio = np.concatenate(self._recording_frames, axis=0)
        return audio

    def write_wav(self, wav_path: str | Path, audio: np.ndarray) -> Path:
        """Write int16 PCM WAV."""
        wav_path = Path(wav_path)
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        if audio.ndim == 1:
            audio = audio[:, None]

        audio_i16 = audio.astype(np.int16, copy=False)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(self._config.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self._config.sample_rate)
            wf.writeframes(audio_i16.tobytes())

        return wav_path

    def read_wav(self, wav_path: str | Path) -> tuple[np.ndarray, int]:
        wav_path = Path(wav_path)
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise ValueError(f"Only 16-bit WAV supported, got sampwidth={sampwidth}")
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
        else:
            audio = audio.reshape(-1, 1)
        return audio, sr

    async def play_wav(
        self,
        wav_path: str | Path,
        *,
        allow_barge_in: bool = True,
        barge_in_timeout_s: float = 30.0,
    ) -> bool:
        """Play a WAV file. Returns True if barge-in interrupted playback."""
        sd = self._require_sounddevice()

        audio, sr = self.read_wav(wav_path)
        audio_f32 = (audio.astype(np.float32) / 32768.0).squeeze(-1)

        interrupted = False

        monitor_stream = None
        if allow_barge_in:

            def monitor_cb(indata, frames, time, status):  # noqa: ANN001
                nonlocal interrupted
                if status:
                    return
                rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
                if rms >= self._config.barge_in_rms_threshold:
                    interrupted = True
                    logger.info(
                        "[VOICE][AUDIO] barge-in triggered rms=%.4f threshold=%.4f",
                        rms,
                        self._config.barge_in_rms_threshold,
                    )
                    try:
                        sd.stop()
                    except Exception:
                        pass

            try:
                monitor_stream = sd.InputStream(
                    samplerate=self._config.sample_rate,
                    channels=self._config.channels,
                    dtype="float32",
                    callback=monitor_cb,
                )
                await asyncio.to_thread(monitor_stream.start)
            except Exception:
                monitor_stream = None

        sd.play(audio_f32, samplerate=sr, blocking=False)

        try:
            await asyncio.wait_for(asyncio.to_thread(sd.wait), timeout=barge_in_timeout_s)
        except asyncio.TimeoutError:
            try:
                sd.stop()
            except Exception:
                pass
        finally:
            if monitor_stream is not None:
                await asyncio.to_thread(monitor_stream.stop)
                await asyncio.to_thread(monitor_stream.close)

        return interrupted
