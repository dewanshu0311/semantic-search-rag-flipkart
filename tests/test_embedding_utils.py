"""Tests for embedding utilities."""
import numpy as np
import pytest


def test_embedding_dimensions():
    """Embeddings from SBERT should be 384-dimensional."""
    from src.embedding_model import EmbeddingModel

    model = EmbeddingModel()
    vectors = model.encode(["hello world", "test sentence"], show_progress=False)
    assert vectors.shape == (2, 384), f"Expected (2, 384), got {vectors.shape}"


def test_embedding_normalization():
    """After normalization, L2 norms should be ~1.0."""
    from src.embedding_model import EmbeddingModel

    model = EmbeddingModel()
    vectors = model.encode(["sample text"], normalize=True, show_progress=False)
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5, f"Norm should be 1.0, got {norm}"


def test_encode_empty_list():
    """Encoding an empty list should return an empty array."""
    from src.embedding_model import EmbeddingModel

    model = EmbeddingModel()
    vectors = model.encode([], show_progress=False)
    assert vectors.shape[0] == 0
