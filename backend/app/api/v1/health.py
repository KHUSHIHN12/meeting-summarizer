"""Service health endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Service health check")
async def health_check() -> dict[str, str]:
    """Return the current API service health status."""
    return {"status": "healthy", "service": "AI Meeting Summarizer", "version": "1.0.0"}
