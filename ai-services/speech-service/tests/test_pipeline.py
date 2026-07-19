"""Unit tests for preprocessing pipeline orchestration."""

from pathlib import Path

import pytest

from pipeline.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.exceptions import FFmpegConversionError, NormalizationError, VADProcessingError
from preprocessing.vad import SpeechRegion, VADResult
from preprocessing.validator import AudioMetadata


class ValidatingStage:
    """Validation boundary returning deterministic source metadata."""

    def validate(self, path: Path, content_type: str | None = None) -> AudioMetadata:
        """Return audio metadata for orchestration tests."""
        return AudioMetadata(12.5, 16_000, 1, "pcm_s16le")


class FailingValidator:
    """Validation boundary that stops the flow immediately."""

    def validate(self, path: Path, content_type: str | None = None) -> AudioMetadata:
        """Raise the original validation failure."""
        raise ValueError("Invalid audio")


class ConverterStage:
    """Conversion boundary with a stable generated WAV path."""

    def __init__(self, output_path: Path, failure: Exception | None = None) -> None:
        self.output_path = output_path
        self.failure = failure

    def convert(self, input_path: Path) -> Path:
        """Return output or fail before later stages run."""
        if self.failure:
            raise self.failure
        return self.output_path


class NormalizerStage:
    """Normalization boundary with a stable WAV path."""

    def __init__(self, output_path: Path, failure: Exception | None = None) -> None:
        self.output_path = output_path
        self.failure = failure

    def normalize(self, input_path: Path) -> Path:
        """Return output or fail before VAD begins."""
        if self.failure:
            raise self.failure
        return self.output_path


class VADStage:
    """VAD boundary with a speech-only WAV path."""

    def __init__(self, output_path: Path, failure: Exception | None = None) -> None:
        self.output_path = output_path
        self.failure = failure

    def detect(self, input_path: Path) -> VADResult:
        """Return VAD output or reproduce a stage failure."""
        if self.failure:
            raise self.failure
        return VADResult(self.output_path, [SpeechRegion(0.0, 12.5)])


def build_pipeline(tmp_path: Path, converter_error: Exception | None = None, normalizer_error: Exception | None = None, vad_error: Exception | None = None) -> PreprocessingPipeline:
    """Build the central preprocessing pipeline with independently injected stages."""
    return PreprocessingPipeline(
        validator=ValidatingStage(),
        converter=ConverterStage(tmp_path / "meeting.wav", converter_error),
        normalizer=NormalizerStage(tmp_path / "meeting_normalized.wav", normalizer_error),
        vad=VADStage(tmp_path / "meeting_vad.wav", vad_error),
    )


def test_preprocessing_pipeline_passes_stage_outputs_sequentially(tmp_path: Path) -> None:
    """Pass original, converted, normalized, and VAD paths in sequence."""
    result = build_pipeline(tmp_path).process(tmp_path / "meeting.mp3")
    assert result.wav_audio.name == "meeting.wav"
    assert result.normalized_audio.name == "meeting_normalized.wav"
    assert result.vad_audio.name == "meeting_vad.wav"
    assert result.sample_rate == 16_000
    assert result.duration == 12.5


def test_invalid_file_stops_at_validation(tmp_path: Path) -> None:
    """Propagate validation failures before conversion begins."""
    pipeline = PreprocessingPipeline(
        validator=FailingValidator(),
        converter=ConverterStage(tmp_path / "unused.wav"),
        normalizer=NormalizerStage(tmp_path / "unused_normalized.wav"),
        vad=VADStage(tmp_path / "unused_vad.wav"),
    )
    with pytest.raises(ValueError, match="Invalid audio"):
        pipeline.process(tmp_path / "invalid.mp3")


def test_ffmpeg_failure_stops_pipeline(tmp_path: Path) -> None:
    """Stop when conversion fails."""
    with pytest.raises(FFmpegConversionError):
        build_pipeline(tmp_path, converter_error=FFmpegConversionError("Conversion failed")).process(tmp_path / "meeting.mp3")


def test_normalization_failure_stops_pipeline(tmp_path: Path) -> None:
    """Stop when normalization fails."""
    with pytest.raises(NormalizationError):
        build_pipeline(tmp_path, normalizer_error=NormalizationError("Normalization failed")).process(tmp_path / "meeting.mp3")


def test_vad_failure_stops_pipeline(tmp_path: Path) -> None:
    """Stop when VAD fails."""
    with pytest.raises(VADProcessingError):
        build_pipeline(tmp_path, vad_error=VADProcessingError("VAD failed")).process(tmp_path / "meeting.mp3")