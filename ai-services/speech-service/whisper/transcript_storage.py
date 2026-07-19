"""Persistent JSON and text storage for completed meeting transcripts."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from whisper.schemas import TranscriptResponse
from preprocessing.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StoredTranscript:
    """Filesystem locations created for a persisted transcript."""

    json_path: Path
    text_path: Path


class TranscriptStorage:
    """Save transcript responses as machine-readable JSON and readable text."""

    def __init__(self, transcripts_directory: Optional[Path] = None) -> None:
        """Configure transcript output below the repository data directory."""
        self._directory = transcripts_directory or (
            Path(__file__).resolve().parents[3] / "data" / "transcripts"
        )

    def save(self, meeting_name: str, transcript: TranscriptResponse) -> StoredTranscript:
        """Create JSON and TXT transcript files for a completed meeting.

        Args:
            meeting_name: Source meeting filename stem used for output naming.
            transcript: Structured transcript returned by the ASR service.

        Returns:
            Filesystem locations for the generated JSON and text outputs.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_filename(meeting_name)
        json_path = self._directory / "{0}_transcript.json".format(safe_name)
        text_path = self._directory / "{0}_transcript.txt".format(safe_name)
        self._write_json(json_path, meeting_name, transcript)
        self._write_text(text_path, meeting_name, transcript)
        logger.info(
            "Transcript files saved",
            extra={"output_path": str(json_path)},
        )
        return StoredTranscript(json_path=json_path, text_path=text_path)

    @staticmethod
    def _safe_filename(value: str) -> str:
        """Create a cross-platform safe transcript basename."""
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return normalized.strip("._") or "meeting"

    @staticmethod
    def _write_json(path: Path, meeting_name: str, transcript: TranscriptResponse) -> None:
        """Persist the exact structured transcript shape requested by clients."""
        payload: dict[str, Any] = {
            "meeting_name": meeting_name,
            "language": transcript.language,
            "duration": transcript.duration,
            "processing_time": transcript.metadata.processing_time,
            "transcript": transcript.transcript,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "start": segment.start_time,
                    "end": segment.end_time,
                    "text": segment.text,
                    "confidence": segment.confidence,
                }
                for segment in transcript.segments
            ],
            "metadata": transcript.metadata.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @staticmethod
    def _write_text(path: Path, meeting_name: str, transcript: TranscriptResponse) -> None:
        """Persist a concise human-readable transcript file."""
        content = (
            "Meeting Name:\n{0}\n\n"
            "Language:\n{1}\n\n"
            "Duration:\n{2:.2f} seconds\n\n"
            "------------------------------------------------\n\n"
            "Transcript\n\n{3}\n\n"
            "------------------------------------------------\n"
        ).format(meeting_name, transcript.language, transcript.duration, transcript.transcript)
        path.write_text(content, encoding="utf-8")
