"""Assembly of segment-level ASR output into a meeting transcript."""

from datetime import datetime, timezone
from typing import Iterable

from whisper.schemas import TranscriptMetadata, TranscriptResponse, TranscriptSegment
from whisper.utils import merge_segments, normalize_language


class TranscriptBuilder:
    """Build a stable API response from Faster-Whisper segment output."""

    def build(
        self,
        meeting_id: str,
        language: str,
        duration: float,
        segments: Iterable[TranscriptSegment],
        processing_time: float,
        model_size: str,
    ) -> TranscriptResponse:
        """Create a complete transcript response and execution metadata."""
        segment_list = list(segments)
        return TranscriptResponse(
            meeting_id=meeting_id,
            language=normalize_language(language),
            duration=duration,
            transcript=merge_segments(segment_list),
            segments=segment_list,
            metadata=TranscriptMetadata(
                processing_time=round(processing_time, 4),
                model_name="faster-whisper",
                model_size=model_size,
                created_at=datetime.now(timezone.utc),
                segment_count=len(segment_list),
                audio_duration=duration,
            ),
        )

