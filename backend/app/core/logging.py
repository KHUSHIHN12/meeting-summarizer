"""Structured application logging configuration."""

import logging
import sys


def configure_logging(log_level: str) -> None:
    """Configure process-wide logging with a production-friendly format."""
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z", stream=sys.stdout, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module."""
    return logging.getLogger(name)
