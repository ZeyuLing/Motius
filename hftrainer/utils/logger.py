"""Logger utility."""

import logging
import os
import sys


_loggers = {}

LOG_FORMAT = '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
LOG_DATEFMT = '%Y/%m/%d %H:%M:%S'


def get_logger(name: str = 'hftrainer', log_level: str = 'INFO') -> logging.Logger:
    """Get or create a logger with the given name."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _loggers[name] = logger
    return logger


def add_file_handler(logger: logging.Logger, filepath: str,
                     log_level: str = 'INFO') -> None:
    """Add a FileHandler to write logs to disk.

    Args:
        logger: The logger to attach the file handler to.
        filepath: Path to the log file.
        log_level: Logging level for the file handler.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    handler = logging.FileHandler(filepath, mode='a')
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
