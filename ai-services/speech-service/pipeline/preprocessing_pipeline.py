"""Orchestration layer for the existing audio preprocessing stages."""

from dataclasses import asdict, dataclass
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
class PreprocessingResult:
    """Typed hand-off object passed from preprocessing to ASR."""

    original_audio: Path
    wav_audio: Path
    normalized_audio: Path
    vad_audio: Path
    duration: float
    sample_rate: int
    processing_time: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation for pipeline clients."""
        payload = asdict(self)
        for key in ("original_audio", "wav_audio", "normalized_audio", "vad_audio"):
            payload[key] = str(payload[key])
        return payload


class PreprocessingPipeline:
    """Coordinate validation, conversion, normalization, and VAD only."""

    def __init__(
        self,
        settings: Optional[AudioProcessingSettings] = None,
        validator: Optional[AudioValidator] = None,
        converter: Optional[FFmpegConverter] = None,
        normalizer: Optional[AudioNormalizer] = None,
        vad: Optional[VoiceActivityDetector] = None,
    ) -> None:
        """Inject existing preprocessing stages without duplicating their logic."""
        active_settings = settings or AudioProcessingSettings()
        self._validator = validator or AudioValidator(active_settings)
        self._converter = converter or FFmpegConverter(active_settings)
        self._normalizer = normalizer or AudioNormalizer(active_settings)
        self._vad = vad or VoiceActivityDetector(active_settings)

    def process(
        self, input_audio_path: str | Path, content_type: Optional[str] = None
    ) -> PreprocessingResult:
        """Run each preprocessing stage using the immediately prior stage output."""
        original_audio = Path(input_audio_path).expanduser().resolve()
        started_at = perf_counter()
        logger.info("Starting preprocessing pipeline", extra={"input_path": str(original_audio)})
        try:
            metadata = self._validator.validate(original_audio, content_type)
            logger.info("Audio validation completed", extra={"input_path": str(original_audio)})

            wav_audio = self._converter.convert(original_audio)
            logger.info("Audio converted", extra={"output_path": str(wav_audio)})

            normalized_audio = self._normalizer.normalize(wav_audio)
            logger.info("Audio normalized", extra={"output_path": str(normalized_audio)})

            vad_result = self._vad.detect(normalized_audio)
            logger.info("Voice Activity Detection completed", extra={"output_path": str(vad_result.output_path)})
        except Exception:
            logger.exception("Preprocessing pipeline failed", extra={"input_path": str(original_audio)})
            raise

        return self._build_result(
            original_audio,
            wav_audio,
            normalized_audio,
            vad_result,
            metadata,
            perf_counter() - started_at,
        )

    @staticmethod
    def _build_result(
        original_audio: Path,
        wav_audio: Path,
        normalized_audio: Path,
        vad_result: VADResult,
        metadata: AudioMetadata,
        processing_time: float,
    ) -> PreprocessingResult:
        """Create the typed result passed to downstream ASR."""
        return PreprocessingResult(
            original_audio=original_audio,
            wav_audio=wav_audio,
            normalized_audio=normalized_audio,
            vad_audio=vad_result.output_path,
            duration=metadata.duration_seconds,
            sample_rate=metadata.sample_rate,
            processing_time=round(processing_time, 4),
        )

