import logging
import sys

import colorlog


def init_logger(log_level: int = logging.INFO, name: str = None) -> logging.Logger:
    """
    Initialize and configure the logger with custom color support for different parts.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        name (str, optional): The name for the logger. If None, root logger is used.
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
    formatter = colorlog.ColoredFormatter(
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

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger
