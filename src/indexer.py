"""
FAISS index manager — create, populate, save, load.

Supported index types: ``flat``, ``hnsw``, ``ivfpq``.

Usage::

    from src.indexer import FAISSIndexer
    idx = FAISSIndexer(dim=384, index_type="hnsw")
    idx.add(embeddings)
    idx.save("data/processed/my_index.faiss")
    idx2 = FAISSIndexer.load("data/processed/my_index.faiss")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """Wrapper around FAISS indexes with create / add / search / save / load."""

    def __init__(
        self,
        dim: int = 384,
        index_type: str = "hnsw",
        cfg: Optional[Config] = None,
    ) -> None:
        self.cfg = cfg or Config()
        self.dim = dim
        self.index_type = index_type.lower()
        self.index: faiss.Index = self._create_index()
        logger.info("Created FAISS index type=%s  dim=%d", self.index_type, dim)

    # ── Factory ────────────────────────────────────────────────────────
    def _create_index(self) -> faiss.Index:
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)

        if self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(self.dim, self.cfg.HNSW_M)
            idx.hnsw.efConstruction = self.cfg.HNSW_EF_CONSTRUCTION
            idx.hnsw.efSearch = self.cfg.HNSW_EF_SEARCH
            return idx

        if self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.cfg.IVF_NLIST,
                self.cfg.PQ_M,
                self.cfg.PQ_NBITS,
            )
            return idx

        raise ValueError(f"Unknown index_type: {self.index_type!r}")

    # ── Add vectors ────────────────────────────────────────────────────
    def add(self, vectors: np.ndarray) -> None:
        """Add normalized vectors to the index (trains IVF if needed)."""
        vectors = vectors.astype(np.float32)
        if self.index_type == "ivfpq" and not self.index.is_trained:
            logger.info("Training IVF-PQ index on %d vectors …", len(vectors))
            self.index.train(vectors)
        t0 = time.perf_counter()
        self.index.add(vectors)
        logger.info("Added %d vectors in %.2f s  (total=%d)",
                     len(vectors), time.perf_counter() - t0, self.index.ntotal)

    # ── Search ─────────────────────────────────────────────────────────
    def search(self, query_vec: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the *k* nearest neighbours.

        Returns
        -------
        distances : np.ndarray, shape (n_queries, k)
        indices   : np.ndarray, shape (n_queries, k)
        """
        query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        return distances, indices

    # ── Persistence ────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Write index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index → %s", path)

    @classmethod
    def load(cls, path: str | Path, dim: int = 384) -> "FAISSIndexer":
        """Load a previously saved index."""
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.cfg = Config()
        obj.index = faiss.read_index(str(path))
        obj.index_type = "loaded"
        logger.info("Loaded FAISS index ← %s  (ntotal=%d)", path, obj.index.ntotal)
        return obj

    @property
    def ntotal(self) -> int:
        return self.index.ntotal
