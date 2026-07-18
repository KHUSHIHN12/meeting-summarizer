"""Tests for real audio validation behavior."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from config import AudioProcessingSettings
from preprocessing.exceptions import CorruptedAudioError, UnsupportedFormatError
from preprocessing.validator import AudioValidator


def test_validator_accepts_valid_wav(tmp_path: Path) -> None:
    """Validate an actual generated 16 kHz WAV file."""
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
    metadata = AudioValidator(AudioProcessingSettings()).validate(audio_path, "audio/wav")
    assert metadata.sample_rate == 16_000


def test_validator_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Reject a non-audio extension before probing it."""
    source = tmp_path / "invalid.txt"
    source.write_text("not audio", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        AudioValidator().validate(source, "text/plain")


def test_validator_rejects_empty_file(tmp_path: Path) -> None:
    """Reject an empty WAV file."""
    source = tmp_path / "empty.wav"
    source.touch()
    with pytest.raises(CorruptedAudioError):
        AudioValidator().validate(source, "audio/wav")
