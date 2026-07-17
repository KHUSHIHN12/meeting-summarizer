"""Business service for meeting upload and metadata operations."""

import asyncio
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings
from app.core.logging import get_logger
from app.repositories.meeting_repository import MeetingRepository
from app.schemas.meeting import MeetingCreate, MeetingResponse

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4"}
ALLOWED_CONTENT_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/m4a", "audio/x-m4a", "video/mp4"}
CHUNK_SIZE_BYTES = 1024 * 1024


class FileValidationError(ValueError):
    """Raised when an uploaded file does not meet upload policy."""


class DuplicateFilenameError(ValueError):
    """Raised when a filename is already registered as a meeting."""


class MeetingNotFoundError(LookupError):
    """Raised when a requested meeting does not exist."""


class MeetingService:
    """Coordinate upload storage and meeting repository operations."""

    def __init__(self, repository: MeetingRepository, settings: Settings) -> None:
        """Initialize the service with persistence and configuration dependencies."""
        self._repository = repository
        self._settings = settings

    async def upload_meeting(self, file: UploadFile, title: str | None, description: str | None) -> MeetingResponse:
        """Validate an upload, store it locally, and create its database record."""
        original_filename = file.filename or ""
        self._validate_file_metadata(original_filename, file.content_type)
        if await self._repository.get_by_original_filename(original_filename):
            raise DuplicateFilenameError("A meeting with this filename already exists.")

        await asyncio.to_thread(self._settings.upload_directory.mkdir, parents=True, exist_ok=True)
        stored_filename = f"{uuid4().hex}{Path(original_filename).suffix.lower()}"
        destination = self._settings.upload_directory / stored_filename
        file_size = 0
        try:
            async with aiofiles.open(destination, "wb") as output:
                while chunk := await file.read(CHUNK_SIZE_BYTES):
                    file_size += len(chunk)
                    if file_size > self._settings.max_upload_size_bytes:
                        raise FileValidationError(f"File exceeds the {self._settings.max_upload_size_bytes} byte upload limit.")
                    await output.write(chunk)

            meeting = await self._repository.create_meeting(MeetingCreate(
                title=title.strip() if title and title.strip() else Path(original_filename).stem,
                description=description.strip() if description and description.strip() else None,
                filename=stored_filename,
                original_filename=original_filename,
                file_size=file_size,
                content_type=file.content_type or "application/octet-stream",
                upload_path=str(destination),
            ))
        except DuplicateKeyError as exc:
            await self._remove_file(destination)
            raise DuplicateFilenameError("A meeting with this filename already exists.") from exc
        except Exception:
            await self._remove_file(destination)
            raise
        finally:
            await file.close()

        logger.info("Meeting '%s' uploaded with id %s", original_filename, meeting.id)
        return meeting

    async def list_meetings(self) -> list[MeetingResponse]:
        """Return all meeting metadata."""
        return await self._repository.list_meetings()

    async def get_meeting(self, meeting_id: str) -> MeetingResponse:
        """Return a meeting or signal that it does not exist."""
        meeting = await self._repository.get_meeting(meeting_id)
        if meeting is None:
            raise MeetingNotFoundError("Meeting not found.")
        return meeting

    async def delete_meeting(self, meeting_id: str) -> None:
        """Delete a meeting record and its associated uploaded file."""
        meeting = await self._repository.delete_meeting(meeting_id)
        if meeting is None:
            raise MeetingNotFoundError("Meeting not found.")
        await self._remove_file(Path(meeting.upload_path))
        logger.info("Meeting '%s' deleted", meeting_id)

    def _validate_file_metadata(self, filename: str, content_type: str | None) -> None:
        """Validate filename extension and MIME type before file persistence."""
        if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise FileValidationError("Only mp3, wav, m4a, and mp4 files are supported.")
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise FileValidationError("The uploaded file has an unsupported MIME type.")

    @staticmethod
    async def _remove_file(path: Path) -> None:
        """Remove a stored upload if it exists."""
        if path.is_file():
            await asyncio.to_thread(path.unlink)
