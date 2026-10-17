"""General utilities for spectra captioning."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure root and library loggers for CLI commands and pipelines.

    By default, shows only WARNING and ERROR messages so standard CLI progress
    stays clean. When ``verbose=True``, enables full DEBUG traces.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ["httpx", "httpcore", "fsspec", "google_genai", "urllib3"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_qualitative_distance(z: float) -> str:
    """Return a qualitative description of distance based on redshift."""
    if z < 0.1:
        return "very close (local universe)"
    elif z < 0.5:
        return "close"
    elif z < 1.0:
        return "intermediate"
    elif z < 2.0:
        return "far"
    else:
        return "very far (high redshift)"

