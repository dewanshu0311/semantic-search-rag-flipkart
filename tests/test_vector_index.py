"""Tests for FAISS index creation, search, save/load."""
import os
import tempfile

import numpy as np
import pytest


def test_flat_index_search():
    """Create Flat index, add random vectors, search returns correct shape."""
    from src.indexer import FAISSIndexer

    dim = 64
    n = 100
    k = 5
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    idx = FAISSIndexer(dim=dim, index_type="flat")
    idx.add(vectors)
    assert idx.ntotal == n

    query = vectors[0:1]
    dists, indices = idx.search(query, k)
    assert dists.shape == (1, k)
    assert indices.shape == (1, k)
    # First result should be the query itself (self-match)
    assert indices[0][0] == 0


def test_hnsw_index_search():
    """HNSW index should return results."""
    from src.indexer import FAISSIndexer

    dim = 32
    n = 50
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    idx = FAISSIndexer(dim=dim, index_type="hnsw")
    idx.add(vectors)
    dists, indices = idx.search(vectors[0:1], 3)
    assert len(indices[0]) == 3


def test_save_load_index():
    """Save and reload a FAISS index — ntotal should match."""
    from src.indexer import FAISSIndexer

    dim = 32
    vectors = np.random.randn(20, dim).astype(np.float32)

    idx = FAISSIndexer(dim=dim, index_type="flat")
    idx.add(vectors)

    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        path = f.name

    try:
        idx.save(path)
        loaded = FAISSIndexer.load(path, dim=dim)
        assert loaded.ntotal == 20
    finally:
        os.unlink(path)
