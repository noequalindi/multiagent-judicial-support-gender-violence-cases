from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = Path(os.getenv("LOG_DIR", "logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "{extra[stage]} | "
            "{message}"
        ),
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_file,
        level=log_level,
        rotation=os.getenv("LOG_ROTATION", "10 MB"),
        retention=os.getenv("LOG_RETENTION", "14 days"),
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
            "{extra[stage]} | {message}"
        ),
    )
    _CONFIGURED = True


def get_logger(stage: str = "app"):
    configure_logging()
    return logger.bind(stage=stage)
