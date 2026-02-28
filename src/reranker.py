"""
Lightweight rating-weighted re-ranker.

Combines semantic similarity score with normalized rating and sentiment
signal to produce a final ranking. Designed to be explainable and
appropriate for a student project (no 600MB cross-encoder download).

Usage::

    from src.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, results, k=5)
"""

from __future__ import annotations

import logging
from typing import List

from src.retriever import SearchResult

logger = logging.getLogger(__name__)

# Sentiment polarity weights
_SENTIMENT_WEIGHT = {"positive": 0.1, "neutral": 0.0, "negative": -0.05}


class Reranker:
    """
    Rating-weighted re-ranker.

    Final score = semantic_score + rating_boost + sentiment_boost

    Parameters
    ----------
    rating_weight : float
        How much to weight the normalized rating (0-1 scale).
    sentiment_weight : float
        How much to weight the sentiment signal.
    """

    def __init__(
        self,
        rating_weight: float = 0.15,
        sentiment_weight: float = 0.10,
    ) -> None:
        self.rating_weight = rating_weight
        self.sentiment_weight = sentiment_weight

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        k: int = 5,
    ) -> List[SearchResult]:
        """
        Re-rank results by combining semantic score with rating and sentiment.

        Parameters
        ----------
        query : str
            The user query (unused in this simple ranker, but kept for API parity).
        results : list[SearchResult]
            Input results from the retriever.
        k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Re-ranked and truncated results.
        """
        if not results:
            return results

        scored = []
        for r in results:
            # Normalize rating to [0, 1]
            rating = float(r.metadata.get("Rate", 3))
            rating_norm = (rating - 1) / 4.0  # maps 1->0, 5->1

            # Sentiment boost
            sentiment = str(r.metadata.get("Sentiment", "neutral")).lower()
            sent_boost = _SENTIMENT_WEIGHT.get(sentiment, 0.0)

            # Combined score
            final_score = (
                r.score
                + self.rating_weight * rating_norm
                + self.sentiment_weight * sent_boost
            )
            scored.append((final_score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for rank, (score, r) in enumerate(scored[:k], start=1):
            reranked.append(SearchResult(
                rank=rank,
                score=score,
                text=r.text,
                metadata=r.metadata,
            ))

        logger.info("Re-ranked %d → %d results for '%s'", len(results), len(reranked), query[:40])
        return reranked
