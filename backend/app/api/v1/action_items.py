"""Action-item endpoint placeholders for future implementation."""

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get("", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def list_action_items() -> None:
    """Reserve the action-item collection endpoint for a future phase."""
    raise HTTPException(status_code=501, detail="Action item operations are not implemented.")
