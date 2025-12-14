"""Local voice subsystem.

This package provides an offline, modular voice I/O channel for the interview
system:

mic -> STT -> orchestrator -> TTS -> speaker

The orchestrator remains the single authority for interview flow and memory.
"""

from recruiting_agent_pollock.voice.stt import (
    STTConfig,
    STTProvider,
    TranscriptionResult,
    WhisperSTT,
)
from recruiting_agent_pollock.voice.tts import PiperTTS, TTSConfig, TTSProvider
from recruiting_agent_pollock.voice.voice_session import VoiceSession, VoiceSessionConfig

__all__ = [
    "STTConfig",
    "STTProvider",
    "TranscriptionResult",
    "WhisperSTT",
    "PiperTTS",
    "TTSConfig",
    "TTSProvider",
    "VoiceSession",
    "VoiceSessionConfig",
]


# Optional audio dependencies (numpy + sounddevice). Keep import-time lightweight.
try:
    from recruiting_agent_pollock.voice.audio_io import AudioIO, AudioIOConfig

    __all__.extend(["AudioIO", "AudioIOConfig"])
except (ModuleNotFoundError, ImportError):
    pass
