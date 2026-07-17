"""FastAPI application factory and lifecycle configuration."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.database.mongodb import mongodb

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize and release shared infrastructure resources."""
    await mongodb.connect(uri=settings.mongodb_uri, database_name=settings.mongodb_database, min_pool_size=settings.mongodb_min_pool_size, max_pool_size=settings.mongodb_max_pool_size, server_selection_timeout_ms=settings.mongodb_server_selection_timeout_ms)
    logger.info("Application startup complete")
    try:
        yield
    finally:
        await mongodb.disconnect()
        logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="Backend API for AI-powered meeting summaries and action items.",
    version=settings.app_version,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Return a consistent response for HTTP exceptions."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a consistent response for request validation errors."""
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors without leaking internal details to clients."""
    logger.exception("Unhandled application error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "An unexpected internal error occurred."})


@app.get("/", tags=["Root"], include_in_schema=False)
async def root() -> dict[str, Any]:
    """Expose a minimal service entry point."""
    return {"service": settings.app_name, "version": settings.app_version}
