"""Orchestrator for the complete audio preparation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from config import AudioProcessingSettings
from preprocessing.ffmpeg_converter import FFmpegConverter
from preprocessing.normalizer import AudioNormalizer
from preprocessing.utils import get_logger
from preprocessing.vad import VADResult, VoiceActivityDetector
from preprocessing.validator import AudioMetadata, AudioValidator

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessedAudio:
    """Final output and metadata from a successful preprocessing run."""

    output_path: Path
    source_metadata: AudioMetadata
    vad_result: VADResult
    processing_time_seconds: float


class AudioProcessor:
    """Execute validation, conversion, normalization, and VAD sequentially."""

    def __init__(self, settings: Optional[AudioProcessingSettings] = None) -> None:
        """Create pipeline components with shared configuration."""
        active_settings = settings or AudioProcessingSettings()
        self._validator = AudioValidator(active_settings)
        self._converter = FFmpegConverter(active_settings)
        self._normalizer = AudioNormalizer(active_settings)
        self._vad = VoiceActivityDetector(active_settings)

    def validate(self, input_path: Path, content_type: Optional[str] = None) -> AudioMetadata:
        """Validate an uploaded audio recording."""
        return self._validator.validate(input_path, content_type)

    def convert(self, input_path: Path) -> Path:
        """Convert audio to standard 16 kHz mono PCM WAV."""
        return self._converter.convert(input_path)

    def normalize(self, input_path: Path) -> Path:
        """Normalize loudness and trim boundary silence."""
        return self._normalizer.normalize(input_path)

    def detect_voice(self, input_path: Path) -> VADResult:
        """Detect speech and remove long silence segments."""
        return self._vad.detect(input_path)

    def process(self, input_path: Path, content_type: Optional[str] = None) -> ProcessedAudio:
        """Run the complete preprocessing pipeline in the required order."""
        started_at = perf_counter()
        try:
            metadata = self.validate(input_path, content_type)
            converted_path = self.convert(input_path)
            normalized_path = self.normalize(converted_path)
            vad_result = self.detect_voice(normalized_path)
        except Exception:
            logger.exception("Audio processing failed", extra={"input_path": str(input_path)})
            raise
        elapsed_seconds = perf_counter() - started_at
        logger.info("Audio processing completed", extra={"input_path": str(input_path), "output_path": str(vad_result.output_path), "elapsed_ms": round(elapsed_seconds * 1000, 2)})
        return ProcessedAudio(vad_result.output_path, metadata, vad_result, elapsed_seconds)
