"""Meeting upload and metadata endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import Settings, get_settings
from app.database.mongodb import get_database
from app.repositories.meeting_repository import MeetingRepository
from app.schemas.meeting import MeetingListResponse, MeetingResponse, UploadMeetingResponse
from app.services.meeting_service import DuplicateFilenameError, FileValidationError, MeetingNotFoundError, MeetingService

router = APIRouter()


def get_meeting_service(database: Annotated[AsyncIOMotorDatabase, Depends(get_database)], settings: Annotated[Settings, Depends(get_settings)]) -> MeetingService:
    """Build a meeting service with request-scoped infrastructure dependencies."""
    return MeetingService(MeetingRepository(database), settings)


@router.post("/upload", response_model=UploadMeetingResponse, status_code=status.HTTP_201_CREATED)
async def upload_meeting(file: Annotated[UploadFile, File(description="Audio or video meeting recording")], title: Annotated[str | None, Form()] = None, description: Annotated[str | None, Form()] = None, service: MeetingService = Depends(get_meeting_service)) -> UploadMeetingResponse:
    """Validate, persist, and register an uploaded meeting recording."""
    try:
        meeting = await service.upload_meeting(file, title, description)
    except FileValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DuplicateFilenameError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return UploadMeetingResponse(meeting_id=meeting.id, status=meeting.status, message="Meeting uploaded successfully")


@router.get("", response_model=MeetingListResponse)
async def list_meetings(service: MeetingService = Depends(get_meeting_service)) -> MeetingListResponse:
    """Return all uploaded meeting metadata ordered by newest first."""
    return MeetingListResponse(items=await service.list_meetings())


@router.get("/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(meeting_id: str, service: MeetingService = Depends(get_meeting_service)) -> MeetingResponse:
    """Return metadata for one meeting."""
    try:
        return await service.get_meeting(meeting_id)
    except MeetingNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting(meeting_id: str, service: MeetingService = Depends(get_meeting_service)) -> None:
    """Delete a meeting's database record and stored upload."""
    try:
        await service.delete_meeting(meeting_id)
    except MeetingNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
