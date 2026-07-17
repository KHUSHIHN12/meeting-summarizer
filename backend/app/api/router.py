"""Central API router composition."""

from fastapi import APIRouter

from app.api.v1 import action_items, health, meetings, summary

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meetings.router, prefix="/meetings", tags=["Meetings"])
api_router.include_router(summary.router, prefix="/summaries", tags=["Summaries"])
api_router.include_router(action_items.router, prefix="/action-items", tags=["Action Items"])
