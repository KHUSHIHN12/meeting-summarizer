"""Health endpoint smoke tests."""

from fastapi.testclient import TestClient
from app.main import app


def test_health_check() -> None:
    """Return the documented health response."""
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "AI Meeting Summarizer", "version": "1.0.0"}
