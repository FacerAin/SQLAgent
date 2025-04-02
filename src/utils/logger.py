import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import colorlog


def init_token_logger(
    log_level: int = logging.INFO, log_to_file: bool = False, log_dir: str = "logs"
) -> logging.Logger:
    """
    Initialize a specialized logger for tracking token usage with LLM API calls.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_file (bool): Whether to log to file in addition to console.
    """
    return init_logger(
        log_level=log_level,
        name="token_usage",
        log_to_file=log_to_file,
        log_dir=log_dir,
    )


def init_logger(
    log_level: int = logging.INFO,
    name: Optional[str] = None,
    log_to_file: bool = False,
    log_dir: str = "logs",
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Initialize and configure the logger with custom color support for different parts.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        name (str, optional): The name for the logger. If None, root logger is used.
        log_to_file (bool): Whether to log to file in addition to console.
        log_dir (str): Directory to store log files.
        max_file_size (int): Maximum file size in bytes before rotating.
        backup_count (int): Number of backup files to keep.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers = []

    # Create console handler and set level to log_level
    ch = colorlog.StreamHandler(sys.stdout)
    ch.setLevel(log_level)

    # Create color formatter with custom colors for different parts
    color_formatter = colorlog.ColoredFormatter(
        "%(asctime)s - %(name_log_color)s%(name)s%(reset)s - %(level_log_color)s%(levelname)s%(reset)s - %(message_log_color)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={
            "name": {
                "DEBUG": "bold_blue",
                "INFO": "bold_blue",
                "WARNING": "bold_blue",
                "ERROR": "bold_blue",
                "CRITICAL": "bold_blue",
            },
            "level": {
                "DEBUG": "purple",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            "message": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "white",
                "ERROR": "white",
                "CRITICAL": "white",
            },
        },
        style="%",
    )

    # Add formatter to console handler
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)

    # Add file handler if requested
    if log_to_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Determine log filename based on logger name
        log_filename = name if name else "root"
        log_file_path = log_path / f"{log_filename}.log"

        # Standard formatter for file logs (no colors)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create rotating file handler
        fh = RotatingFileHandler(
            log_file_path, maxBytes=max_file_size, backupCount=backup_count
        )
        fh.setLevel(log_level)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger
