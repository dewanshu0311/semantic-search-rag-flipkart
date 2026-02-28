"""
Text preprocessing pipeline.

Usage::

    from src.preprocessing import preprocess_series
    clean = preprocess_series(df["Text"])
"""

from __future__ import annotations

import logging
import re
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def remove_html(text: str) -> str:
    """Strip HTML tags."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Strip URLs."""
    return re.sub(r"https?://\S+", "", text)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single string."""
    text = str(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = collapse_whitespace(text)
    return text.lower()


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply preprocessing to an entire pandas Series."""
    result = series.astype(str).apply(preprocess_text)
    logger.info("Preprocessed %d texts", len(result))
    return result
