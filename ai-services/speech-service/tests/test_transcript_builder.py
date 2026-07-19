"""Unit tests for transcript assembly helpers."""

from whisper.schemas import TranscriptSegment
from whisper.transcript_builder import TranscriptBuilder
from whisper.utils import format_timestamp, merge_segments, normalize_language


def test_builder_merges_segments_into_transcript() -> None:
    """Build one response with normalized language and combined text."""
    response = TranscriptBuilder().build(
        meeting_id="meeting-1",
        language="English",
        duration=5.0,
        segments=[
            TranscriptSegment(segment_id=0, start_time=0, end_time=2, text="Hello"),
            TranscriptSegment(segment_id=1, start_time=2, end_time=5, text="world"),
        ],
        processing_time=0.4,
        model_size="small",
    )
    assert response.language == "en"
    assert response.transcript == "Hello world"
    assert response.metadata.segment_count == 2


def test_utility_helpers_format_and_merge_text() -> None:
    """Return stable timestamp, language, and transcript formatting."""
    assert format_timestamp(65.125) == "00:01:05.125"
    assert normalize_language("Hindi") == "hi"
    assert merge_segments([]) == ""
