"""Tests for automatic transcript file persistence."""

import json
from datetime import datetime, timezone
from pathlib import Path

from whisper.schemas import TranscriptMetadata, TranscriptResponse, TranscriptSegment
from whisper.transcript_storage import TranscriptStorage


def build_transcript() -> TranscriptResponse:
    """Create a complete transcript response for filesystem tests."""
    return TranscriptResponse(
        meeting_id="meeting-1",
        language="en",
        duration=12.5,
        transcript="Good morning everyone.",
        segments=[
            TranscriptSegment(
                segment_id=0,
                start_time=0.0,
                end_time=2.5,
                text="Good morning everyone.",
                confidence=0.9,
            )
        ],
        metadata=TranscriptMetadata(
            processing_time=0.4,
            model_name="faster-whisper",
            model_size="small",
            created_at=datetime.now(timezone.utc),
            segment_count=1,
            audio_duration=12.5,
        ),
    )


def test_storage_creates_json_and_text_transcripts(tmp_path: Path) -> None:
    """Create the output folder and persist both requested file formats."""
    stored = TranscriptStorage(tmp_path / "transcripts").save("sample meeting", build_transcript())
    assert stored.json_path.is_file()
    assert stored.text_path.is_file()
    payload = json.loads(stored.json_path.read_text(encoding="utf-8"))
    assert payload["meeting_name"] == "sample meeting"
    assert payload["segments"][0]["start"] == 0.0
    assert "Meeting Name:\nsample meeting" in stored.text_path.read_text(encoding="utf-8")
