"""Sequential orchestration pipeline for speech audio preprocessing."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AudioProcessingSettings
from preprocessing.ffmpeg_converter import FFmpegConverter
from preprocessing.normalizer import AudioNormalizer
from preprocessing.utils import get_logger
from preprocessing.vad import VADResult, VoiceActivityDetector
from preprocessing.validator import AudioMetadata, AudioValidator

logger = get_logger(__name__)
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PreprocessingResult:
    """Structured output produced by a successful preprocessing pipeline run."""

    success: bool
    original_audio: str
    wav_audio: str
    normalized_audio: str
    vad_audio: str
    processing_time: float
    sample_rate: int
    duration: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the pipeline result."""
        return asdict(self)


class AudioPreprocessingPipeline:
    """Run validation, conversion, normalization, and VAD in strict sequence.

    Dependencies can be supplied explicitly to support isolated integration tests
    and future graph-node adapters without coupling stages to the pipeline.
    """

    def __init__(
        self,
        settings: Optional[AudioProcessingSettings] = None,
        validator: Optional[AudioValidator] = None,
        converter: Optional[FFmpegConverter] = None,
        normalizer: Optional[AudioNormalizer] = None,
        vad: Optional[VoiceActivityDetector] = None,
    ) -> None:
        """Initialize the pipeline and its preprocessing stage dependencies."""
        active_settings = settings or AudioProcessingSettings()
        self._validator = validator or AudioValidator(active_settings)
        self._converter = converter or FFmpegConverter(active_settings)
        self._normalizer = normalizer or AudioNormalizer(active_settings)
        self._vad = vad or VoiceActivityDetector(active_settings)

    def process(
        self,
        input_audio_path: str | Path,
        content_type: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PreprocessingResult:
        """Process one recording through every required preprocessing stage.

        Args:
            input_audio_path: Source MP3, WAV, M4A, or MP4 recording.
            content_type: Optional upload MIME type to validate.
            progress_callback: Optional callback invoked after each successful stage.

        Returns:
            A structured result with all generated paths and source metadata.

        Raises:
            AudioProcessingError: Any stage-specific preprocessing exception.
        """
        source_path = Path(input_audio_path).expanduser().resolve()
        started_at = perf_counter()
        logger.info("Starting preprocessing pipeline", extra={"input_path": str(source_path)})

        try:
            metadata = self._validator.validate(source_path, content_type)
            self._report("Audio validated", progress_callback)

            wav_path = self._converter.convert(source_path)
            self._report("Audio converted to WAV", progress_callback)

            normalized_path = self._normalizer.normalize(wav_path)
            self._report("Audio normalized", progress_callback)

            vad_result = self._vad.detect(normalized_path)
            self._report("Voice Activity Detection completed", progress_callback)
        except Exception:
            logger.exception("Preprocessing pipeline failed", extra={"input_path": str(source_path)})
            raise

        processing_time = perf_counter() - started_at
        result = self._build_result(
            source_path, wav_path, normalized_path, vad_result, metadata, processing_time
        )
        logger.info(
            "Pipeline finished successfully",
            extra={
                "input_path": str(source_path),
                "output_path": result.vad_audio,
                "elapsed_ms": round(processing_time * 1000, 2),
            },
        )
        self._report("Pipeline finished successfully", progress_callback)
        return result

    @staticmethod
    def _report(message: str, callback: Optional[ProgressCallback]) -> None:
        """Emit CLI progress only when a callback is supplied."""
        if callback is not None:
            callback(message)

    @staticmethod
    def _build_result(
        source_path: Path,
        wav_path: Path,
        normalized_path: Path,
        vad_result: VADResult,
        metadata: AudioMetadata,
        processing_time: float,
    ) -> PreprocessingResult:
        """Assemble the externally consumable preprocessing result."""
        return PreprocessingResult(
            success=True,
            original_audio=str(source_path),
            wav_audio=str(wav_path),
            normalized_audio=str(normalized_path),
            vad_audio=str(vad_result.output_path),
            processing_time=round(processing_time, 4),
            sample_rate=metadata.sample_rate,
            duration=metadata.duration_seconds,
        )


def main() -> int:
    """Run the preprocessing pipeline from a terminal command."""
    parser = argparse.ArgumentParser(description="Prepare meeting audio for Whisper ASR.")
    parser.add_argument("input_audio_path", type=Path, help="Path to the uploaded recording")
    parser.add_argument("--content-type", default=None, help="Optional upload MIME type")
    arguments = parser.parse_args()

    try:
        result = AudioPreprocessingPipeline().process(
            arguments.input_audio_path,
            content_type=arguments.content_type,
            progress_callback=lambda message: print("INFO  {0}".format(message)),
        )
    except Exception as exc:
        print("ERROR {0}".format(exc), file=sys.stderr)
        return 1

    print(result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
