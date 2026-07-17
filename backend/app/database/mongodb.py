"""MongoDB connection lifecycle management."""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.logging import get_logger

logger = get_logger(__name__)


class MongoDBConnection:
    """Manage the asynchronous MongoDB client for the application lifecycle."""

    def __init__(self) -> None:
        """Create an uninitialized database connection manager."""
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None

    async def connect(self, uri: str, database_name: str) -> None:
        """Create the MongoDB client and select its configured database."""
        self._client = AsyncIOMotorClient(uri)
        self._database = self._client[database_name]
        logger.info("MongoDB client configured for database '%s'", database_name)

    async def disconnect(self) -> None:
        """Close the MongoDB client when the application stops."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB client closed")

    @property
    def database(self) -> AsyncIOMotorDatabase:
        """Return the active database or raise if startup has not completed."""
        if self._database is None:
            raise RuntimeError("MongoDB has not been initialized.")
        return self._database


mongodb = MongoDBConnection()
