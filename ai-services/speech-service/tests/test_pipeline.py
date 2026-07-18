"""Unit tests for preprocessing pipeline orchestration."""

from pathlib import Path

import pytest

from preprocessing.exceptions import FFmpegConversionError, NormalizationError, VADProcessingError
from preprocessing.pipeline import AudioPreprocessingPipeline
from preprocessing.vad import SpeechRegion, VADResult
from preprocessing.validator import AudioMetadata


class ValidatingStage:
    """Small test stage that returns deterministic metadata."""

    def validate(self, path: Path, content_type: str | None = None) -> AudioMetadata:
        """Return real-shaped audio metadata for pipeline orchestration tests."""
        return AudioMetadata(12.5, 16_000, 1, "pcm_s16le")


class FailingValidator:
    """Test stage that models invalid input without invoking later stages."""

    def validate(self, path: Path, content_type: str | None = None) -> AudioMetadata:
        """Signal that the input file cannot be validated."""
        raise ValueError("Invalid audio")


class ConverterStage:
    """Test conversion stage that returns a material intermediate path."""

    def __init__(self, output_path: Path, failure: Exception | None = None) -> None:
        self.output_path = output_path
        self.failure = failure

    def convert(self, input_path: Path) -> Path:
        """Return conversion output or reproduce an FFmpeg failure."""
        if self.failure:
            raise self.failure
        return self.output_path


class NormalizerStage:
    """Test normalization stage that returns a material intermediate path."""

    def __init__(self, output_path: Path, failure: Exception | None = None) -> None:
        self.output_path = output_path
        self.failure = failure

    def normalize(self, input_path: Path) -> Path:
        """Return normalized output or reproduce a normalization failure."""
        if self.failure:
            raise self.failure
        return self.output_path


class VADStage:
    """Test VAD stage that returns a speech-cleaned audio artifact."""

    def __init__(self, output_path: Path, failure: Exception | None = None) -> None:
        self.output_path = output_path
        self.failure = failure

    def detect(self, input_path: Path) -> VADResult:
        """Return VAD output or reproduce a VAD failure."""
        if self.failure:
            raise self.failure
        return VADResult(self.output_path, [SpeechRegion(0.0, 12.5)])


def build_pipeline(tmp_path: Path, converter_error: Exception | None = None, normalizer_error: Exception | None = None, vad_error: Exception | None = None) -> AudioPreprocessingPipeline:
    """Build a pipeline with concrete stage contracts and controlled artifacts."""
    return AudioPreprocessingPipeline(
        validator=ValidatingStage(),
        converter=ConverterStage(tmp_path / "meeting.wav", converter_error),
        normalizer=NormalizerStage(tmp_path / "meeting_normalized.wav", normalizer_error),
        vad=VADStage(tmp_path / "meeting_vad.wav", vad_error),
    )


def test_successful_pipeline_execution(tmp_path: Path) -> None:
    """Pass each stage output to the next and return all final paths."""
    result = build_pipeline(tmp_path).process(tmp_path / "meeting.mp3")
    assert result.success is True
    assert result.wav_audio.endswith("meeting.wav")
    assert result.normalized_audio.endswith("meeting_normalized.wav")
    assert result.vad_audio.endswith("meeting_vad.wav")
    assert result.sample_rate == 16_000
    assert result.duration == 12.5


def test_invalid_file_stops_at_validation(tmp_path: Path) -> None:
    """Propagate validation errors before conversion begins."""
    pipeline = AudioPreprocessingPipeline(
        validator=FailingValidator(),
        converter=ConverterStage(tmp_path / "unused.wav"),
        normalizer=NormalizerStage(tmp_path / "unused_normalized.wav"),
        vad=VADStage(tmp_path / "unused_vad.wav"),
    )
    with pytest.raises(ValueError, match="Invalid audio"):
        pipeline.process(tmp_path / "invalid.mp3")


def test_ffmpeg_failure_stops_pipeline(tmp_path: Path) -> None:
    """Propagate conversion failures without starting normalization or VAD."""
    with pytest.raises(FFmpegConversionError):
        build_pipeline(tmp_path, converter_error=FFmpegConversionError("Conversion failed")).process(tmp_path / "meeting.mp3")


def test_normalization_failure_stops_pipeline(tmp_path: Path) -> None:
    """Propagate normalization failures without starting VAD."""
    with pytest.raises(NormalizationError):
        build_pipeline(tmp_path, normalizer_error=NormalizationError("Normalization failed")).process(tmp_path / "meeting.mp3")


def test_vad_failure_stops_pipeline(tmp_path: Path) -> None:
    """Propagate VAD failures after successful preceding stages."""
    with pytest.raises(VADProcessingError):
        build_pipeline(tmp_path, vad_error=VADProcessingError("VAD failed")).process(tmp_path / "meeting.mp3")
