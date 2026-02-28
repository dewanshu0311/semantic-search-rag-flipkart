"""
Central configuration for the Semantic Search & RAG System.

All tunable parameters live here so notebooks and scripts can do::

    from src.config import Config
    cfg = Config()
    print(cfg.EMBEDDING_DIM)  # 384
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Immutable project configuration."""

    # ── Paths ──────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_RAW: Path = field(init=False)
    DATA_PROCESSED: Path = field(init=False)

    # ── Dataset ────────────────────────────────────────────────────────
    # Flipkart Product Reviews Dataset
    DATASET_CSV: str = "Dataset-SA.csv"
    # Column mapping for Flipkart schema
    COL_PRODUCT: str = "product_name"
    COL_PRICE: str = "product_price"
    COL_RATING: str = "Rate"
    COL_REVIEW: str = "Review"
    COL_SUMMARY: str = "Summary"
    COL_SENTIMENT: str = "Sentiment"

    # ── Sampling ───────────────────────────────────────────────────────
    SAMPLE_ONLY: bool = field(
        default_factory=lambda: os.getenv("SAMPLE_ONLY", "true").lower() == "true"
    )
    SAMPLE_SIZE: int = 5_000
    FULL_SIZE: int = 100_000

    # ── Embedding ──────────────────────────────────────────────────────
    SBERT_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    USE_OPENAI: bool = field(default_factory=lambda: bool(os.getenv("OPENAI_API_KEY")))
    OPENAI_MODEL: str = "text-embedding-3-small"

    # ── FAISS ──────────────────────────────────────────────────────────
    DEFAULT_INDEX: str = "hnsw"      # "flat", "hnsw", "ivfpq"
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 128
    IVF_NLIST: int = 100
    PQ_M: int = 48
    PQ_NBITS: int = 8

    # ── Retrieval ──────────────────────────────────────────────────────
    TOP_K: int = 10
    HYBRID_ALPHA: float = 0.5        # 0=pure BM25, 1=pure dense

    # ── RAG / LLM ─────────────────────────────────────────────────────
    LLM_PROVIDER: str = field(
        default_factory=lambda: "openai" if os.getenv("OPENAI_API_KEY") else "local"
    )
    LLM_MODEL: str = "gpt-3.5-turbo"

    def __post_init__(self) -> None:
        self.DATA_RAW = self.PROJECT_ROOT / "data" / "raw"
        self.DATA_PROCESSED = self.PROJECT_ROOT / "data" / "processed"
        self.DATA_RAW.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        n = self.SAMPLE_SIZE if self.SAMPLE_ONLY else self.FULL_SIZE
        logger.info(
            "Config loaded — SAMPLE_ONLY=%s  rows=%d  index=%s",
            self.SAMPLE_ONLY, n, self.DEFAULT_INDEX,
        )

    @property
    def n_rows(self) -> int:
        """Number of rows to load."""
        return self.SAMPLE_SIZE if self.SAMPLE_ONLY else self.FULL_SIZE
