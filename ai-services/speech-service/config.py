"""Configuration for the speech preprocessing service."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioProcessingSettings:
    """Runtime settings for the audio preprocessing pipeline."""

    processed_directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "processed"
    )
    max_file_size_bytes: int = 500 * 1024 * 1024
    max_duration_seconds: float = 4 * 60 * 60
    min_sample_rate: int = 8_000
    max_sample_rate: int = 192_000
    target_sample_rate: int = 16_000
    vad_aggressiveness: int = 2
    vad_frame_duration_ms: int = 30
    max_silence_between_speech_ms: int = 300

