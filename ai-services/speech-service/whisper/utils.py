"""Reusable helpers for Whisper transcript processing."""

from typing import Iterable, Optional

from whisper.schemas import TranscriptSegment

LANGUAGE_ALIASES = {
    "english": "en",
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "german": "de",
}


def format_timestamp(seconds: float) -> str:
    """Format seconds as an HH:MM:SS.mmm timestamp."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_part, milliseconds_part = divmod(remainder, 1000)
    return "{0:02d}:{1:02d}:{2:02d}.{3:03d}".format(
        hours, minutes, seconds_part, milliseconds_part
    )


def normalize_language(language: Optional[str]) -> str:
    """Normalize Whisper language labels and codes to lowercase ISO-like codes."""
    normalized = (language or "unknown").strip().lower()
    return LANGUAGE_ALIASES.get(normalized, normalized)


def merge_segments(segments: Iterable[TranscriptSegment]) -> str:
    """Merge non-empty segment text into one whitespace-normalized transcript."""
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def calculate_processing_statistics(
    processing_time: float, duration: float, segment_count: int
) -> dict[str, float | int]:
    """Calculate reusable transcription processing statistics."""
    return {
        "processing_time": round(processing_time, 4),
        "audio_duration": round(duration, 4),
        "segment_count": segment_count,
        "realtime_factor": round(processing_time / duration, 4) if duration else 0.0,
    }

