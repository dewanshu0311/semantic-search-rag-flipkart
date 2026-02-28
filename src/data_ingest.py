"""
Data ingestion for Flipkart Product Reviews.

Handles loading, cleaning, and enriching the Flipkart dataset.
Reviews in this dataset are very short (~12 chars avg), so we
combine Summary + Review into a single ``combined_text`` column
that gives embeddings richer signal.

Usage::

    from src.data_ingest import load_flipkart
    df = load_flipkart(cfg)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config import Config
from src.preprocessing import preprocess_series

logger = logging.getLogger(__name__)


def load_flipkart(cfg: Optional[Config] = None) -> pd.DataFrame:
    """
    Load and prepare the Flipkart Product Reviews dataset.

    Steps:
        1. Read CSV from ``data/raw/``.
        2. Drop duplicates and rows with missing text.
        3. Normalize the ``Sentiment`` column.
        4. Create ``combined_text`` = Summary + " " + Review.
        5. Clean text via preprocessing pipeline.
        6. Sample to ``cfg.n_rows`` for speed control.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with ``combined_text`` ready for embedding.
    """
    cfg = cfg or Config()
    csv_path = cfg.DATA_RAW / cfg.DATASET_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Please place Dataset-SA.csv in data/raw/."
        )

    logger.info("Loading Flipkart dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw dataset: %d rows × %d cols", *df.shape)

    # ── Drop duplicates & missing ──────────────────────────────────────
    df = df.drop_duplicates()
    for col in [cfg.COL_SUMMARY, cfg.COL_REVIEW]:
        df[col] = df[col].fillna("")

    # ── Normalize sentiment ────────────────────────────────────────────
    df[cfg.COL_SENTIMENT] = (
        df[cfg.COL_SENTIMENT]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    valid_sentiments = {"positive", "negative", "neutral"}
    df.loc[~df[cfg.COL_SENTIMENT].isin(valid_sentiments), cfg.COL_SENTIMENT] = "neutral"

    # ── Combine text columns ───────────────────────────────────────────
    # Reviews are very short (~12 chars avg), so we enrich with Summary
    df["combined_text"] = (
        df[cfg.COL_SUMMARY].astype(str).str.strip()
        + " "
        + df[cfg.COL_REVIEW].astype(str).str.strip()
    ).str.strip()

    # Drop rows where combined text is empty
    df = df[df["combined_text"].str.len() > 2].reset_index(drop=True)

    # ── Clean text ─────────────────────────────────────────────────────
    df["combined_text"] = preprocess_series(df["combined_text"])

    # ── Sample ─────────────────────────────────────────────────────────
    n = min(cfg.n_rows, len(df))
    if n < len(df):
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
        logger.info("Sampled %d rows (SAMPLE_ONLY=%s)", n, cfg.SAMPLE_ONLY)
    else:
        logger.info("Using full dataset: %d rows", len(df))

    # Ensure Rating is numeric
    df[cfg.COL_RATING] = pd.to_numeric(df[cfg.COL_RATING], errors="coerce").fillna(3)

    logger.info(
        "Dataset ready: %d rows | Products: %d | Sentiment dist: %s",
        len(df),
        df[cfg.COL_PRODUCT].nunique(),
        df[cfg.COL_SENTIMENT].value_counts().to_dict(),
    )
    return df


def get_product_names(df: pd.DataFrame, cfg: Optional[Config] = None) -> list:
    """Return sorted list of unique product names."""
    cfg = cfg or Config()
    return sorted(df[cfg.COL_PRODUCT].unique().tolist())
