"""
Utility helpers: timing, chunking, serialization.

Usage::

    from src.utils import chunk_text, timed

    @timed
    def slow_fn():
        ...

    chunks = chunk_text("long text ...", size=256, overlap=50)
"""

from __future__ import annotations

import logging
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd

logger = logging.getLogger(__name__)


# ── Timing decorator ──────────────────────────────────────────────────
def timed(fn: Callable) -> Callable:
    """Decorator that logs wall-clock time of a function call."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("%s completed in %.2f s", fn.__name__, elapsed)
        return result

    return wrapper


# ── Text chunking ─────────────────────────────────────────────────────
def chunk_text(text: str, size: int = 256, overlap: int = 50) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.

    Parameters
    ----------
    text : str
        Input text.
    size : int
        Maximum characters per chunk.
    overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# ── CSV loader ────────────────────────────────────────────────────────
def read_csv(path: Path | str, n_rows: int | None = None) -> pd.DataFrame:
    """Read a CSV, optionally limiting to *n_rows*."""
    df = pd.read_csv(path, nrows=n_rows)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ── Pickle helpers ────────────────────────────────────────────────────
def save_pickle(obj: Any, path: Path | str) -> None:
    """Serialize *obj* to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved pickle → %s", path)


def load_pickle(path: Path | str) -> Any:
    """Deserialize a pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded pickle ← %s", path)
    return obj
