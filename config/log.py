# config/log.py

import logging
import sys

from config.settings import DEBUG_MODE

# Plugin-style modules (ultralytics / torchreid) are noisy at DEBUG level.
_DEFAULT_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO
_configured: set = set()

logging.basicConfig(
    level=_DEFAULT_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a shared logger for the given module name.
    Level is driven once from DEBUG_MODE in config/settings.py.
    """
    logger = logging.getLogger(f"people_counting.{name}")
    if name not in _configured:
        logger.setLevel(_DEFAULT_LEVEL)
        logger.propagate = False
        _configured.add(name)
    return logger