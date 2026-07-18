"""Integration skeleton for WebRTC voice activity detection."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("webrtcvad")

from config import AudioProcessingSettings
from preprocessing.exceptions import VADProcessingError
from preprocessing.vad import VoiceActivityDetector


def test_vad_rejects_silence_only_wav(tmp_path: Path) -> None:
    """Reject a real silence-only WAV because it contains no speech frames."""
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    detector = VoiceActivityDetector(AudioProcessingSettings(processed_directory=tmp_path))
    with pytest.raises(VADProcessingError):
        detector.detect(source)