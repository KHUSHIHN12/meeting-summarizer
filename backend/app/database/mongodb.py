"""MongoDB connection lifecycle and dependency management."""

from collections.abc import AsyncIterator
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

    async def connect(self, uri: str, database_name: str, min_pool_size: int = 1, max_pool_size: int = 20, server_selection_timeout_ms: int = 5_000) -> None:
        """Connect to MongoDB Atlas and configure the meetings collection index."""
        self._client = AsyncIOMotorClient(uri, minPoolSize=min_pool_size, maxPoolSize=max_pool_size, serverSelectionTimeoutMS=server_selection_timeout_ms, retryWrites=True)
        self._database = self._client[database_name]
        await self._client.admin.command("ping")
        await self._database.meetings.create_index("original_filename", unique=True)
        logger.info("Connected to MongoDB database '%s'", database_name)

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


async def get_database() -> AsyncIterator[AsyncIOMotorDatabase]:
    """Provide the initialized MongoDB database to API dependencies."""
    yield mongodb.database
