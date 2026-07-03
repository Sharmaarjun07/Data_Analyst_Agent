"""
utils/logger.py

Initializes logging once and hands out the same configured logger everywhere
else in the project, instead of every module calling
`logging.basicConfig(...)` (and risking duplicate handlers / inconsistent
formats).

Usage
-----
    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")

Log level is controlled centrally via utils.config.config.LOG_LEVEL (which
in turn can be overridden with the AUTOML_LOG_LEVEL environment variable).
"""

from __future__ import annotations

import logging
import sys

from utils.config import config

_CONFIGURED_ROOT = "automl"
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_root_once() -> None:
    """Attach a single StreamHandler to the shared root logger, idempotently."""
    root = logging.getLogger(_CONFIGURED_ROOT)
    if root.handlers:
        return  # already configured — avoid duplicate log lines

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    root.propagate = False


def get_logger(name: str = _CONFIGURED_ROOT) -> logging.Logger:
    """
    Return a logger namespaced under 'automl' (e.g. 'automl.services.prediction'
    when called as get_logger(__name__) from within the automl package), sharing
    the single configured handler and level set up on first call.
    """
    _configure_root_once()

    if name == _CONFIGURED_ROOT:
        return logging.getLogger(_CONFIGURED_ROOT)

    child_name = name if name.startswith(f"{_CONFIGURED_ROOT}.") else f"{_CONFIGURED_ROOT}.{name}"
    return logging.getLogger(child_name)


def set_level(level: str) -> None:
    """Change the log level at runtime for every logger under the shared root, e.g. set_level('DEBUG')."""
    _configure_root_once()
    logging.getLogger(_CONFIGURED_ROOT).setLevel(getattr(logging, level.upper(), logging.INFO))
