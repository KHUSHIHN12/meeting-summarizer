"""Pydantic schemas for meeting metadata and API responses."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MeetingStatus(str,Enum):
    """Lifecycle states for a meeting recording."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MeetingCreate(BaseModel):
    """Data required to create a persisted meeting record."""

    title: str
    description: str | None = None
    filename: str
    original_filename: str
    file_size: int = Field(ge=0)
    content_type: str
    upload_path: str
    status: MeetingStatus = MeetingStatus.UPLOADED


class MeetingResponse(MeetingCreate):
    """Public representation of a meeting record."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime


class MeetingListResponse(BaseModel):
    """Collection response for meeting metadata."""

    items: list[MeetingResponse]


class UploadMeetingResponse(BaseModel):
    """Response returned after successfully uploading a meeting."""

    meeting_id: str
    status: MeetingStatus
    message: str
