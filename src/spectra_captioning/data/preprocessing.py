"""Preprocessing utilities for cleaning quotes before captioning.

We previously stripped object names here, but that is now handled natively
by instructing the LLM to generalize the text.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def clean_quotes(quotes: list[str]) -> list[str]:
    """Clean and prepare quotes for captioning.

    Steps:
    1. Remove empty or trivially short quotes (< 20 characters).
    2. Deduplicate exact string matches.

    Args:
        quotes: Raw quotes extracted from the dataframe.

    Returns:
        Cleaned, deduplicated quotes ready for the prompt.
    """
    cleaned: list[str] = []
    seen: set[str] = set()

    for quote in quotes:
        quote = quote.strip()
        
        # Remove quotes that are too short to be useful.
        if len(quote) < 20:
            continue

        # Deduplicate.
        if quote not in seen:
            seen.add(quote)
            cleaned.append(quote)

    logger.debug(
        "Cleaned %d quotes -> %d after filtering short quotes and dedup.",
        len(quotes),
        len(cleaned),
    )
    return cleaned
