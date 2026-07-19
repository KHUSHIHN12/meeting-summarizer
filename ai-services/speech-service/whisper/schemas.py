"""Pydantic response models for Whisper transcription."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """One timestamped portion of recognized speech."""

    segment_id: int = Field(ge=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    text: str
    confidence: Optional[float] = None


class TranscriptMetadata(BaseModel):
    """Execution details associated with a transcript."""

    model_config = ConfigDict(from_attributes=True)

    processing_time: float = Field(ge=0)
    model_name: str
    model_size: str
    created_at: datetime
    segment_count: int = Field(ge=0)
    audio_duration: float = Field(ge=0)


class TranscriptResponse(BaseModel):
    """Complete ASR result for a VAD-processed meeting recording."""

    meeting_id: str
    language: str
    duration: float = Field(ge=0)
    transcript: str
    segments: list[TranscriptSegment]
    metadata: TranscriptMetadata

