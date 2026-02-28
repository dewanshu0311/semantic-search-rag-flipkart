"""
Dense retriever — wraps FAISS search with metadata lookup.

Supports optional product filtering for product-aware search.

Usage::

    from src.retriever import DenseRetriever
    retriever = DenseRetriever(indexer, emb, texts, metadata)
    results = retriever.query("good battery life", k=5, product_filter="Smartwatch")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.embedding_model import EmbeddingModel
from src.indexer import FAISSIndexer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    rank: int
    score: float
    text: str
    metadata: Dict[str, Any]


class DenseRetriever:
    """Dense vector retriever over a FAISS index."""

    def __init__(
        self,
        indexer: FAISSIndexer,
        embedding_model: EmbeddingModel,
        texts: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.texts = texts
        self.metadata = metadata

    def query(
        self,
        query_text: str,
        k: int = 5,
        product_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Retrieve top-K results for a query.

        Parameters
        ----------
        query_text : str
            The user query.
        k : int
            Number of results to return.
        product_filter : str, optional
            If provided, only return results from this product.

        Returns
        -------
        list[SearchResult]
        """
        q_vec = self.embedding_model.encode(
            [query_text], normalize=True, show_progress=False
        )

        # If product_filter is set, retrieve more candidates then filter
        fetch_k = k * 5 if product_filter else k

        distances, indices = self.indexer.search(q_vec, fetch_k)

        results: List[SearchResult] = []
        rank = 1
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else {}

            # Apply product filter
            if product_filter:
                prod = meta.get("product_name", "")
                if product_filter.lower() not in str(prod).lower():
                    continue

            results.append(SearchResult(
                rank=rank,
                score=float(dist),
                text=self.texts[idx],
                metadata=meta,
            ))
            rank += 1

            if len(results) >= k:
                break

        logger.info(
            "Query '%s' → %d results (filter=%s)",
            query_text[:50], len(results), product_filter,
        )
        return results
