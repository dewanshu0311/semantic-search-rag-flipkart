"""
Hybrid search: BM25 (sparse) + FAISS (dense) score fusion.

Combines keyword matching with semantic similarity using tunable alpha.
Uses ``rank_bm25`` if available, falls back to scikit-learn TF-IDF.

Usage::

    from src.hybrid_search import HybridSearcher
    hybrid = HybridSearcher(retriever, texts, metadata, alpha=0.6)
    results = hybrid.query("energy efficient AC", k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.retriever import DenseRetriever, SearchResult

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search fusing BM25 and dense retrieval scores.

    Parameters
    ----------
    dense_retriever : DenseRetriever
        The FAISS-backed retriever.
    texts : list[str]
        Corpus texts (same order as the FAISS index).
    metadata : list[dict]
        Metadata per document.
    alpha : float
        Fusion weight: 0 = pure BM25, 1 = pure dense.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        alpha: float = 0.5,
    ) -> None:
        self.dense = dense_retriever
        self.texts = texts
        self.metadata = metadata
        self.alpha = alpha
        self._bm25 = None
        self._tfidf_matrix = None
        self._tfidf_vectorizer = None
        self._init_sparse()

    def _init_sparse(self) -> None:
        """Build sparse retriever (BM25 or TF-IDF fallback)."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in self.texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built on %d documents", len(self.texts))
        except ImportError:
            logger.warning("rank_bm25 not found — falling back to TF-IDF")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(max_features=10_000)
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self.texts)
            logger.info("TF-IDF index built on %d documents", len(self.texts))

    def _sparse_scores(self, query: str) -> np.ndarray:
        """Get sparse relevance scores for all docs."""
        if self._bm25 is not None:
            tokens = query.lower().split()
            return self._bm25.get_scores(tokens)
        else:
            q_vec = self._tfidf_vectorizer.transform([query])
            return (self._tfidf_matrix @ q_vec.T).toarray().flatten()

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def query(
        self,
        query_text: str,
        k: int = 5,
        product_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.

        Returns top-K results ranked by:
            final_score = alpha * dense_norm + (1 - alpha) * sparse_norm
        """
        # Dense scores for all docs via full index search
        q_vec = self.dense.embedding_model.encode(
            [query_text], normalize=True, show_progress=False
        )
        n_total = len(self.texts)
        distances, indices = self.dense.indexer.search(q_vec, min(n_total, 500))

        # Build dense score map
        dense_map: Dict[int, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < n_total:
                dense_map[idx] = float(dist)

        # Sparse scores
        sparse_scores = self._sparse_scores(query_text)

        # Candidate set: union of dense top + sparse top
        sparse_top = np.argsort(sparse_scores)[::-1][:500]
        candidates = set(dense_map.keys()) | set(sparse_top.tolist())

        # Score fusion
        scored: List[tuple] = []
        for idx in candidates:
            d_score = dense_map.get(idx, 0.0)
            s_score = sparse_scores[idx] if idx < len(sparse_scores) else 0.0
            scored.append((idx, d_score, s_score))

        if not scored:
            return []

        # Normalize within candidate set
        d_arr = np.array([s[1] for s in scored])
        s_arr = np.array([s[2] for s in scored])
        d_norm = self._normalize(d_arr)
        s_norm = self._normalize(s_arr)

        final_scored = []
        for i, (idx, _, _) in enumerate(scored):
            final = self.alpha * d_norm[i] + (1 - self.alpha) * s_norm[i]
            final_scored.append((final, idx))

        final_scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        rank = 1
        for score, idx in final_scored:
            if idx >= len(self.texts):
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else {}

            if product_filter:
                prod = str(meta.get("product_name", ""))
                if product_filter.lower() not in prod.lower():
                    continue

            results.append(SearchResult(
                rank=rank,
                score=score,
                text=self.texts[idx],
                metadata=meta,
            ))
            rank += 1
            if len(results) >= k:
                break

        logger.info(
            "Hybrid query '%s' → %d results (α=%.2f)",
            query_text[:40], len(results), self.alpha,
        )
        return results
