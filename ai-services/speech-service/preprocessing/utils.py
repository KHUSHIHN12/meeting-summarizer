"""Shared helpers for preprocessing components."""

import json
import logging
from pathlib import Path
from uuid import uuid4


class JsonFormatter(logging.Formatter):
    """Render log records as compact JSON for machine-readable observability."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the core log record and supported structured attributes."""
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attribute in ("input_path", "output_path", "duration_seconds", "elapsed_ms"):
            if hasattr(record, attribute):
                payload[attribute] = getattr(record, attribute)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a structured stream handler once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def build_output_path(directory: Path, source_path: Path, stage: str) -> Path:
    """Create a collision-resistant WAV path for a processing stage."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{0}_{1}_{2}.wav".format(source_path.stem, stage, uuid4().hex)

