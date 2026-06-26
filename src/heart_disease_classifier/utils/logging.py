"""
heart_disease_classifier.utils.logging
=======================================
Centralised logging configuration for the package.
Call :func:`setup_logging` once at the start of your script or CLI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure the root logger for the package.

    Parameters
    ----------
    level : int
        Logging level, e.g. ``logging.DEBUG``. Default ``INFO``.
    log_file : str or Path, optional
        If given, log messages are also written to this file.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
