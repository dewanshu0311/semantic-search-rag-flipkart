"""
Visualization helpers for the Semantic Search & RAG System.

Provides PCA/UMAP dimensionality reduction and rich Flipkart-specific
visualizations: product-colored scatter, sentiment heatmaps, rating
distributions, and word clouds.

Usage::

    from src.visualization import plot_embeddings_2d, plot_product_dashboard
    fig = plot_embeddings_2d(vectors, labels=scores, method='pca')
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Color palette for products ────────────────────────────────────────
PRODUCT_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#d35400", "#8e44ad",
]


def _reduce(vectors: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
    """Reduce embedding dimensions for plotting."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42).fit_transform(vectors)
    elif method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=n_components, random_state=42).fit_transform(vectors)
        except ImportError:
            logger.warning("umap-learn not installed — falling back to PCA")
            return _reduce(vectors, "pca", n_components)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_embeddings_2d(
    vectors: np.ndarray,
    labels: Optional[Sequence] = None,
    method: str = "pca",
    title: str = "Embedding Visualization",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    2D scatter plot of embeddings colored by labels.

    Parameters
    ----------
    vectors : np.ndarray
        Shape (N, D) embedding matrix.
    labels : sequence, optional
        Categorical labels for coloring.
    method : str
        "pca" or "umap".
    title : str
        Plot title.

    Returns
    -------
    matplotlib.Figure
    """
    coords = _reduce(vectors, method)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = np.array([l == label for l in labels])
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[cmap(i)], label=str(label), alpha=0.6, s=15,
            )
        ax.legend(fontsize=8, markerscale=2, loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=10, c="steelblue")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} Dim 1")
    ax.set_ylabel(f"{method.upper()} Dim 2")
    plt.tight_layout()
    return fig


def plot_rating_distribution(ratings: Sequence, figsize: tuple = (8, 5)) -> plt.Figure:
    """Bar chart of rating distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    unique, counts = np.unique(ratings, return_counts=True)
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax.bar(unique, counts, color=colors[:len(unique)], edgecolor="black", alpha=0.85)
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (⭐)")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_sentiment_distribution(sentiments: Sequence, figsize: tuple = (8, 5)) -> plt.Figure:
    """Pie chart of sentiment distribution."""
    import pandas as pd
    series = pd.Series(sentiments)
    counts = series.value_counts()
    colors = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=[colors.get(s, "#95a5a6") for s in counts.index],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 11},
    )
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_product_rating_heatmap(
    df,
    product_col: str = "product_name",
    rating_col: str = "Rate",
    sentiment_col: str = "Sentiment",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Heatmap: average rating per product × sentiment.
    """
    import pandas as pd
    pivot = df.pivot_table(
        values=rating_col,
        index=product_col,
        columns=sentiment_col,
        aggfunc="mean",
    )
    # Shorten product names for readability
    pivot.index = [n[:30] + "..." if len(n) > 30 else n for n in pivot.index]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, vmin=1, vmax=5,
    )
    ax.set_title("Average Rating by Product × Sentiment", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_text_length_distribution(texts: Sequence[str], figsize: tuple = (10, 5)) -> plt.Figure:
    """Histogram of text lengths."""
    lengths = [len(str(t)) for t in texts]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(np.median(lengths), color="red", linestyle="--",
               label=f"Median: {np.median(lengths):.0f}")
    ax.set_title("Text Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_wordcloud(texts: Sequence[str], title: str = "Word Cloud", figsize: tuple = (12, 6)):
    """Generate a word cloud from texts."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed — pip install wordcloud")
        return None

    all_text = " ".join(str(t) for t in texts)
    wc = WordCloud(
        width=800, height=400, background_color="white",
        colormap="viridis", max_words=150,
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
