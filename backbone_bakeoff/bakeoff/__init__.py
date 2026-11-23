"""EuroSAT backbone bake-off: data, models, benchmarking."""
import logging
import sys

logging.getLogger(__name__).addHandler(logging.NullHandler())  # library default: silent unless configured


class _StdoutHandler(logging.StreamHandler):
    """Writes to current sys.stdout (Jupyter / pytest swap it after handler creation)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _):
        pass


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """bakeoff.* records -> stdout as bare messages (CLI mains + notebook). Idempotent."""
    logger = logging.getLogger(__name__)
    if not any(isinstance(handler, _StdoutHandler) for handler in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # root handlers (e.g. Lightning, basicConfig) -> no duplicate lines
    return logger
