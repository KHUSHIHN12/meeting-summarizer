"""Validation for uploaded recordings before preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ffmpeg

from config import AudioProcessingSettings
from preprocessing.exceptions import (
    AudioTooLargeError,
    CorruptedAudioError,
    InvalidSampleRateError,
    RecordingTooLongError,
    UnsupportedFormatError,
)
from preprocessing.utils import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4"}
ALLOWED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4",
    "audio/m4a", "audio/x-m4a", "video/mp4",
}


@dataclass(frozen=True)
class AudioMetadata:
    """Validated metadata extracted from the audio stream."""

    duration_seconds: float
    sample_rate: int
    channels: int
    codec_name: str


class AudioValidator:
    """Validate file policy and probe audio streams with FFmpeg."""

    def __init__(self, settings: Optional[AudioProcessingSettings] = None) -> None:
        """Initialize the validator with processing limits."""
        self._settings = settings or AudioProcessingSettings()

    def validate(self, path: Path, content_type: Optional[str] = None) -> AudioMetadata:
        """Validate an uploaded audio recording and return its metadata.

        Args:
            path: Path to the uploaded file.
            content_type: Optional MIME type provided by the upload client.

        Raises:
            UnsupportedFormatError: For unsupported extensions or MIME types.
            AudioTooLargeError: If the file exceeds the configured size.
            CorruptedAudioError: If FFmpeg cannot read an audio stream.
        """
        path = path.resolve()
        self._validate_file_policy(path, content_type)
        probe_data = self._probe(path)
        metadata = self._extract_metadata(probe_data)
        self._validate_metadata(metadata)
        logger.info("Audio validation completed", extra={"input_path": str(path), "duration_seconds": metadata.duration_seconds})
        return metadata

    def _validate_file_policy(self, path: Path, content_type: Optional[str]) -> None:
        """Validate filesystem state, extension, MIME type, and size."""
        if not path.is_file() or path.stat().st_size == 0:
            raise CorruptedAudioError("The uploaded audio file is missing or empty.")
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError("Supported formats are MP3, WAV, M4A, and MP4.")
        if content_type is not None and content_type.lower() not in ALLOWED_MIME_TYPES:
            raise UnsupportedFormatError("The uploaded audio MIME type is not supported.")
        if path.stat().st_size > self._settings.max_file_size_bytes:
            raise AudioTooLargeError("The uploaded audio file exceeds the maximum size.")

    @staticmethod
    def _probe(path: Path) -> dict[str, Any]:
        """Probe audio metadata with FFmpeg and translate probe failures."""
        try:
            return ffmpeg.probe(str(path))
        except ffmpeg.Error as exc:
            raise CorruptedAudioError("FFmpeg could not read the uploaded audio file.") from exc
        except OSError as exc:
            raise CorruptedAudioError("FFmpeg is unavailable or the audio file cannot be read.") from exc

    @staticmethod
    def _extract_metadata(probe_data: dict[str, Any]) -> AudioMetadata:
        """Extract the first valid audio stream from FFmpeg probe output."""
        stream = next((item for item in probe_data.get("streams", []) if item.get("codec_type") == "audio"), None)
        if stream is None:
            raise CorruptedAudioError("The uploaded file does not contain an audio stream.")
        try:
            return AudioMetadata(
                duration_seconds=float(probe_data.get("format", {}).get("duration", 0)),
                sample_rate=int(stream["sample_rate"]),
                channels=int(stream["channels"]),
                codec_name=str(stream.get("codec_name", "unknown")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptedAudioError("The uploaded audio has incomplete metadata.") from exc

    def _validate_metadata(self, metadata: AudioMetadata) -> None:
        """Validate duration and source sample rate constraints."""
        if metadata.duration_seconds <= 0:
            raise CorruptedAudioError("The uploaded audio has no playable duration.")
        if not self._settings.min_sample_rate <= metadata.sample_rate <= self._settings.max_sample_rate:
            raise InvalidSampleRateError("The audio sample rate is outside the supported range.")
        if metadata.duration_seconds > self._settings.max_duration_seconds:
            raise RecordingTooLongError("The uploaded recording exceeds the maximum duration.")

