from __future__ import annotations

import logging
import os
from typing import Optional


DEFAULT_FORMAT = "[%(levelname)s] %(asctime)s %(name)s - %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging level and format.

    Level can be provided explicitly or via LOG_LEVEL env var.
    """

    env_level = os.getenv("LOG_LEVEL")
    raw_level = level or env_level or "INFO"

    try:
        numeric_level = logging.getLevelName(raw_level.upper())
        if isinstance(numeric_level, str):
            raise ValueError
    except Exception:
        numeric_level = logging.INFO

    logging.basicConfig(level=numeric_level, format=DEFAULT_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
