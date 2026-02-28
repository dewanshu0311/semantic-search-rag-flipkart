"""Smoke test for the RAG pipeline on toy data."""
import numpy as np
import pytest


def test_rag_pipeline_smoke():
    """End-to-end: build a tiny index, run RAG, assert answer is non-empty."""
    from src.embedding_model import EmbeddingModel
    from src.indexer import FAISSIndexer
    from src.retriever import DenseRetriever
    from src.reranker import Reranker
    from src.rag_pipeline import RAGPipeline

    texts = [
        "This air cooler is amazing, works great in summer heat.",
        "Terrible product, stopped working after one day.",
        "Great smartwatch for fitness tracking, very comfortable.",
        "The sound quality is terrible and battery drains fast.",
        "Best value for money, highly recommend this product.",
    ]
    metadata = [
        {"Rate": 5, "Summary": "Amazing cooler", "Sentiment": "positive", "product_name": "Air Cooler"},
        {"Rate": 1, "Summary": "Terrible product", "Sentiment": "negative", "product_name": "Air Cooler"},
        {"Rate": 5, "Summary": "Great smartwatch", "Sentiment": "positive", "product_name": "Smartwatch"},
        {"Rate": 1, "Summary": "Bad quality", "Sentiment": "negative", "product_name": "Speaker"},
        {"Rate": 5, "Summary": "Best value", "Sentiment": "positive", "product_name": "Speaker"},
    ]

    emb = EmbeddingModel()
    vectors = emb.encode(texts, normalize=True, show_progress=False)

    indexer = FAISSIndexer(dim=vectors.shape[1], index_type="flat")
    indexer.add(vectors)

    retriever = DenseRetriever(indexer, emb, texts, metadata)
    reranker = Reranker()
    rag = RAGPipeline(retriever, reranker, top_k=3, rerank_k=2)

    answer = rag.answer("Which product is best for summer?")
    assert len(answer) > 10, "RAG answer should be non-trivial"
    assert "cooler" in answer.lower() or "summer" in answer.lower() or "amazing" in answer.lower(), \
        "Answer should reference the relevant air cooler review"
