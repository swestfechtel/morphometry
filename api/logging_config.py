"""Logging configuration for the API.

Replaces the previous ``utils.init_logger`` which created a new log file named
with ``time.asctime()`` (spaces/colons, a fresh file every restart). Here we use
a single stable, rotating log file plus console output.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(levelname)s: %(asctime)s - %(name)s - %(funcName)s - %(message)s"


def configure_logging(log_dir: Path, level: str = "INFO", name: str = "api") -> logging.Logger:
    """Configure and return the application logger.

    :param log_dir: Directory for the rotating log file (created if missing).
    :param level: Logging level name (e.g. 'INFO', 'DEBUG').
    :param name: Logger name.
    :return: The configured logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # idempotent across reloads / worker forks
        file_handler = RotatingFileHandler(log_dir / "api.log", maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(console)

    return logger
