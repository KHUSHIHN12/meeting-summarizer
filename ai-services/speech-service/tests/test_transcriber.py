"""Unit tests for the Faster-Whisper transcription service."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from whisper.exceptions import AudioNotFoundError, TranscriptionError, UnsupportedAudioError
from whisper.model import WhisperSettings
from whisper.transcriber import TranscriptionService


class FakeModel:
    """Deterministic ASR model implementing Faster-Whisper's transcribe contract."""

    def transcribe(self, audio_path: str, vad_filter: bool) -> tuple[list[object], object]:
        """Return transcript segments without external inference."""
        return (
            [
                SimpleNamespace(id=0, start=0.0, end=1.0, text=" Hello", avg_logprob=-0.2),
                SimpleNamespace(id=1, start=1.0, end=2.0, text=" world", avg_logprob=-0.1),
            ],
            SimpleNamespace(language="en"),
        )


class FakeLoader:
    """Loader boundary for testing transcription independently of model download."""

    settings = WhisperSettings("small", "cpu", "int8")

    def get_model(self) -> FakeModel:
        """Return a deterministic inference dependency."""
        return FakeModel()


def create_wav(path: Path) -> Path:
    """Create a valid VAD-shaped PCM WAV fixture."""
    sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    return path


def test_successful_transcription(tmp_path: Path) -> None:
    """Convert model segments into a complete transcript response."""
    response = TranscriptionService(FakeLoader()).transcribe("meeting-1", create_wav(tmp_path / "vad.wav"))
    assert response.transcript == "Hello world"
    assert len(response.segments) == 2


def test_missing_audio_raises_not_found(tmp_path: Path) -> None:
    """Reject a missing VAD output before attempting model inference."""
    with pytest.raises(AudioNotFoundError):
        TranscriptionService(FakeLoader()).transcribe("meeting-1", tmp_path / "missing.wav")


def test_non_wav_audio_is_rejected(tmp_path: Path) -> None:
    """Accept only the WAV output expected from the preprocessing pipeline."""
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"not a WAV")
    with pytest.raises(UnsupportedAudioError):
        TranscriptionService(FakeLoader()).transcribe("meeting-1", source)


def test_corrupted_wav_raises_transcription_error(tmp_path: Path) -> None:
    """Reject a WAV file that cannot be decoded by soundfile."""
    source = tmp_path / "corrupted.wav"
    source.write_bytes(b"not a valid WAV")
    with pytest.raises(TranscriptionError):
        TranscriptionService(FakeLoader()).transcribe("meeting-1", source)
