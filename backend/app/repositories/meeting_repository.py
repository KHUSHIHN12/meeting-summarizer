"""MongoDB repository for meeting records."""

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from app.schemas.meeting import MeetingCreate, MeetingResponse, MeetingStatus


class MeetingRepository:
    """Encapsulate MongoDB persistence operations for meetings."""

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        """Initialize the repository with an injected database."""
        self._collection = database.get_collection("meetings")

    async def create_meeting(self, meeting: MeetingCreate) -> MeetingResponse:
        """Insert a meeting document and return its persisted representation."""
        now = datetime.now(UTC)
        document = meeting.model_dump(mode="json") | {"created_at": now, "updated_at": now}
        result = await self._collection.insert_one(document)
        document["_id"] = result.inserted_id
        return self._to_response(document)

    async def get_meeting(self, meeting_id: str) -> MeetingResponse | None:
        """Find one meeting by its public identifier."""
        if not ObjectId.is_valid(meeting_id):
            return None
        document = await self._collection.find_one({"_id": ObjectId(meeting_id)})
        return self._to_response(document) if document else None

    async def get_by_original_filename(self, filename: str) -> MeetingResponse | None:
        """Find an existing meeting by its original uploaded filename."""
        document = await self._collection.find_one({"original_filename": filename})
        return self._to_response(document) if document else None

    async def list_meetings(self) -> list[MeetingResponse]:
        """Return all meeting documents ordered newest first."""
        cursor = self._collection.find({}).sort("created_at", -1)
        return [self._to_response(document) async for document in cursor]

    async def update_status(self, meeting_id: str, status: MeetingStatus) -> MeetingResponse | None:
        """Set a meeting processing status and refresh its modification timestamp."""
        if not ObjectId.is_valid(meeting_id):
            return None
        document = await self._collection.find_one_and_update(
            {"_id": ObjectId(meeting_id)},
            {"$set": {"status": status.value, "updated_at": datetime.now(UTC)}},
            return_document=ReturnDocument.AFTER,
        )
        return self._to_response(document) if document else None

    async def delete_meeting(self, meeting_id: str) -> MeetingResponse | None:
        """Delete and return a meeting record by identifier."""
        if not ObjectId.is_valid(meeting_id):
            return None
        document = await self._collection.find_one_and_delete({"_id": ObjectId(meeting_id)})
        return self._to_response(document) if document else None

    @staticmethod
    def _to_response(document: dict[str, Any]) -> MeetingResponse:
        """Map a MongoDB document to the public schema."""
        return MeetingResponse(id=str(document["_id"]), **{key: value for key, value in document.items() if key != "_id"})
