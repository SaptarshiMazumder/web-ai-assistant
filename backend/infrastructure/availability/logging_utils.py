"""Availability check logging - writes to file and stdout for debugging."""

import logging
import os
import sys
from typing import Optional

AVAILABILITY_LOGGER = "availability"

_configured = False


def get_logger() -> logging.Logger:
    global _configured
    logger = logging.getLogger(AVAILABILITY_LOGGER)
    if not _configured:
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("[availability] %(message)s"))
            logger.addHandler(ch)
        _configured = True
    return logger


def setup_availability_file_logging(screenshots_dir: str) -> None:
    """Add file handler so availability logs also go to availability_debug.log."""
    if not screenshots_dir:
        return
    os.makedirs(screenshots_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(screenshots_dir, "availability_debug.log"))
    logger = get_logger()
    # Remove existing file handlers to avoid duplicates if task retries
    for h in list(logger.handlers):
        try:
            if getattr(h, "baseFilename", "") == log_path:
                logger.removeHandler(h)
        except Exception:
            pass
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)


def teardown_availability_file_logging(screenshots_dir: Optional[str]) -> None:
    """Remove file handler after task completes."""
    if not screenshots_dir:
        return
    log_path = os.path.abspath(os.path.join(screenshots_dir, "availability_debug.log"))
    logger = get_logger()
    for h in list(logger.handlers):
        try:
            if getattr(h, "baseFilename", "") == log_path:
                logger.removeHandler(h)
                break
        except Exception:
            pass
