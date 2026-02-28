"""
Embedding model wrapper — supports SBERT and optional OpenAI.

Usage::

    from src.embedding_model import EmbeddingModel
    emb = EmbeddingModel()
    vectors = emb.encode(["hello world", "foo bar"])
    assert vectors.shape == (2, 384)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Unified embedding interface for SBERT and OpenAI."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self._model = None
        self._provider: str = "openai" if self.cfg.USE_OPENAI else "sbert"
        logger.info("EmbeddingModel provider=%s", self._provider)

    # ── lazy load ──────────────────────────────────────────────────────
    def _load_sbert(self):
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            logger.info("Loading SBERT model: %s", self.cfg.SBERT_MODEL)
            self._model = SentenceTransformer(self.cfg.SBERT_MODEL)
        return self._model

    # ── encode ─────────────────────────────────────────────────────────
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        normalize : bool
            L2-normalize output vectors (required for cosine-via-IP).
        show_progress : bool
            Show SBERT progress bar.
        batch_size : int
            Encoding batch size.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), EMBEDDING_DIM)``.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self._provider == "openai":
            return self._encode_openai(texts, normalize)
        return self._encode_sbert(texts, normalize, show_progress, batch_size)

    def _encode_sbert(
        self, texts: List[str], normalize: bool, show_progress: bool, batch_size: int
    ) -> np.ndarray:
        model = self._load_sbert()
        embeddings = model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        if normalize:
            import faiss
            faiss.normalize_L2(embeddings)
        logger.info("Encoded %d texts → (%d, %d)", len(texts), *embeddings.shape)
        return embeddings.astype(np.float32)

    def _encode_openai(self, texts: List[str], normalize: bool) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  to use OpenAI embeddings")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(input=texts, model=self.cfg.OPENAI_MODEL)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        logger.info("OpenAI encoded %d texts → (%d, %d)", len(texts), *vecs.shape)
        return vecs

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        if self._provider == "openai":
            return 1536
        return self.cfg.EMBEDDING_DIM
