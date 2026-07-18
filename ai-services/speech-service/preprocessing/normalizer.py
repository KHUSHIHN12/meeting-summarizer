"""Loudness, clipping, and boundary-silence normalization."""

from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment, effects, silence

from config import AudioProcessingSettings
from preprocessing.exceptions import NormalizationError
from preprocessing.utils import build_output_path, get_logger

logger = get_logger(__name__)


class AudioNormalizer:
    """Normalize a converted WAV before voice activity detection."""

    def __init__(self, settings: Optional[AudioProcessingSettings] = None) -> None:
        """Initialize normalization settings."""
        self._settings = settings or AudioProcessingSettings()

    def normalize(self, input_path: Path) -> Path:
        """Trim boundary silence, reduce clipping, and normalize peak loudness."""
        output_path = build_output_path(self._settings.processed_directory, input_path, "normalized")
        try:
            audio = AudioSegment.from_wav(input_path)
            audio = self._trim_boundary_silence(audio)
            audio = self._reduce_clipping(audio)
            if audio.rms == 0:
                raise NormalizationError("The converted audio contains only silence.")
            effects.normalize(audio, headroom=1.0).export(output_path, format="wav")
        except NormalizationError:
            output_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise NormalizationError("Audio loudness normalization failed.") from exc
        logger.info("Audio normalization completed", extra={"input_path": str(input_path), "output_path": str(output_path)})
        return output_path

    @staticmethod
    def _trim_boundary_silence(audio: AudioSegment) -> AudioSegment:
        """Remove leading and trailing non-speech silence while preserving content."""
        threshold = max(audio.dBFS - 16, -50) if audio.dBFS != float("-inf") else -50
        regions = silence.detect_nonsilent(audio, min_silence_len=300, silence_thresh=threshold)
        if not regions:
            return audio
        start, end = regions[0][0], regions[-1][1]
        return audio[start:end]

    @staticmethod
    def _reduce_clipping(audio: AudioSegment) -> AudioSegment:
        """Soft-limit clipped samples before peak normalization."""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if samples.size == 0:
            return audio
        peak = float(np.max(np.abs(samples)))
        full_scale = float(1 << (8 * audio.sample_width - 1))
        if peak < full_scale * 0.995:
            return audio
        limited = np.tanh(samples / full_scale) * (full_scale - 1)
        raw = limited.astype(audio.array_type).tobytes()
        return audio._spawn(raw)

