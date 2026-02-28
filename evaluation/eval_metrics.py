"""
Evaluation metrics: Precision@K, Recall@K, MRR.

Usage::

    from evaluation.eval_metrics import precision_at_k, recall_at_k
    p = precision_at_k(retrieved_ids=[1,2,3,4,5], relevant_ids={2,4}, k=5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """
    Precision@K: fraction of top-K retrieved items that are relevant.

    Parameters
    ----------
    retrieved_ids : list[int]
        Ordered list of retrieved document IDs.
    relevant_ids : set[int]
        Ground truth relevant IDs.
    k : int

    Returns
    -------
    float
        Value in [0, 1].
    """
    topk = retrieved_ids[:k]
    hits = len(set(topk) & relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """
    Recall@K: fraction of all relevant items found in top-K.

    Parameters
    ----------
    retrieved_ids : list[int]
    relevant_ids : set[int]
    k : int

    Returns
    -------
    float
    """
    topk = retrieved_ids[:k]
    hits = len(set(topk) & relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first relevant item.

    Returns
    -------
    float
    """
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_queries(
    queries: List[Dict],
    search_fn,
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a batch of queries against ground truth.

    Parameters
    ----------
    queries : list[dict]
        Each dict has ``"query"`` (str) and ``"relevant_ids"`` (set[int]).
    search_fn : callable
        Function(query_text, k) → List[SearchResult].
    k : int

    Returns
    -------
    dict
        ``{"precision@k": ..., "recall@k": ..., "mrr": ...}``
    """
    p_scores, r_scores, mrr_scores = [], [], []
    for q in queries:
        results = search_fn(q["query"], k)
        rids = [r.metadata.get("Id", r.metadata.get("id", i)) for i, r in enumerate(results)]
        rel = set(q["relevant_ids"])
        p_scores.append(precision_at_k(rids, rel, k))
        r_scores.append(recall_at_k(rids, rel, k))
        mrr_scores.append(mrr(rids, rel))

    metrics = {
        f"precision@{k}": sum(p_scores) / len(p_scores),
        f"recall@{k}": sum(r_scores) / len(r_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
    }
    logger.info("Evaluation results: %s", metrics)
    return metrics
