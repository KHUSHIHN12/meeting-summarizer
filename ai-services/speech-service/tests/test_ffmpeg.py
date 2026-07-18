"""Integration skeleton for FFmpeg conversion."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from config import AudioProcessingSettings
from preprocessing.ffmpeg_converter import FFmpegConverter


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg binary is required")
def test_converter_outputs_standard_wav(tmp_path: Path) -> None:
    """Convert a real stereo WAV to the required mono 16 kHz format."""
    source = tmp_path / "stereo.wav"
    sf.write(source, np.zeros((44_100, 2), dtype=np.float32), 44_100)
    result = FFmpegConverter(AudioProcessingSettings(processed_directory=tmp_path)).convert(source)
    audio, sample_rate = sf.read(result)
    assert sample_rate == 16_000
    assert audio.ndim == 1
