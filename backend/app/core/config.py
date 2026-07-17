"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the backend service."""

    app_name: str = "AI Meeting Summarizer"
    app_version: str = "1.0.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_database: str = "meeting_summarizer"
    mongodb_min_pool_size: int = Field(default=1, ge=0)
    mongodb_max_pool_size: int = Field(default=20, ge=1)
    mongodb_server_selection_timeout_ms: int = Field(default=5_000, ge=1_000)
    upload_directory: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "uploads")
    max_upload_size_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore", enable_decoding=False)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Accept either a comma-separated string or a list of origins."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("upload_directory", mode="before")
    @classmethod
    def resolve_upload_directory(cls, value: str | Path) -> Path:
        """Resolve the upload directory to an absolute local path."""
        return Path(value).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance for the application process."""
    return Settings()
