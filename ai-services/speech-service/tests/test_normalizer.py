"""Integration skeleton for audio normalization."""

from pathlib import Path

import numpy as np
import soundfile as sf

from config import AudioProcessingSettings
from preprocessing.normalizer import AudioNormalizer


def test_normalizer_trims_boundary_silence(tmp_path: Path) -> None:
    """Trim generated leading and trailing silence from a real WAV file."""
    source = tmp_path / "silence.wav"
    signal = np.concatenate((np.zeros(8_000), np.full(16_000, 0.2), np.zeros(8_000)))
    sf.write(source, signal, 16_000)
    result = AudioNormalizer(AudioProcessingSettings(processed_directory=tmp_path)).normalize(source)
    normalized, _ = sf.read(result)
    assert len(normalized) < len(signal)
