"""
End-to-end RAG pipeline for Flipkart Product Reviews.

Retrieve → Rerank → Generate answers grounded in product review evidence.

Usage::

    from src.rag_pipeline import RAGPipeline
    rag = RAGPipeline(dense_retriever, reranker)
    answer = rag.answer("Which smartwatch has the best battery life?")
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import List, Optional

from src.retriever import DenseRetriever, SearchResult
from src.reranker import Reranker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful Flipkart product review analyst.
Answer the user's question using ONLY the retrieved product review excerpts below.
Cite the product name and rating when relevant.
If the reviews don't contain enough information, say so honestly.

Retrieved Product Reviews:
{context}
"""


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for product reviews."""

    def __init__(
        self,
        retriever: DenseRetriever,
        reranker: Optional[Reranker] = None,
        top_k: int = 10,
        rerank_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_k = rerank_k
        self._provider = "openai" if os.getenv("OPENAI_API_KEY") else "local"
        logger.info("RAGPipeline initialized — provider=%s", self._provider)

    def retrieve(
        self,
        query: str,
        product_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Step 1+2: retrieve then optionally rerank."""
        results = self.retriever.query(query, k=self.top_k, product_filter=product_filter)
        if self.reranker:
            results = self.reranker.rerank(query, results, k=self.rerank_k)
        return results

    @staticmethod
    def _build_context(results: List[SearchResult]) -> str:
        """Format retrieved reviews as LLM context."""
        parts = []
        for r in results:
            product = r.metadata.get("product_name", "Unknown Product")
            rating = r.metadata.get("Rate", "?")
            sentiment = r.metadata.get("Sentiment", "unknown")
            parts.append(
                f"[Product: {product} | Rating: {rating}/5 | Sentiment: {sentiment}]\n"
                f"{r.text[:500]}"
            )
        return "\n---\n".join(parts)

    def answer(
        self,
        query: str,
        product_filter: Optional[str] = None,
    ) -> str:
        """
        Full RAG: retrieve context and generate an answer.

        Falls back to a local extractive summary if no OpenAI key is set.
        """
        results = self.retrieve(query, product_filter=product_filter)
        context = self._build_context(results)

        if self._provider == "openai":
            return self._call_openai(query, context)
        return self._simulate_local(query, results, context)

    def _call_openai(self, query: str, context: str) -> str:
        """Call OpenAI API for generation."""
        try:
            from openai import OpenAI
        except ImportError:
            return self._simulate_local(query, [], context)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ]
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=512
        )
        answer = resp.choices[0].message.content
        logger.info("OpenAI generated %d chars", len(answer))
        return answer

    @staticmethod
    def _simulate_local(
        query: str,
        results: List[SearchResult],
        context: str,
    ) -> str:
        """
        Local extractive summary — no API key needed.

        Synthesizes an answer from retrieved reviews showing product,
        rating, and sentiment for each source.
        """
        if not results:
            return "No relevant product reviews found for your query."

        lines = [f"Based on {len(results)} retrieved product reviews:\n"]
        for r in results[:5]:
            product = r.metadata.get("product_name", "Unknown")
            rating = r.metadata.get("Rate", "?")
            sentiment = r.metadata.get("Sentiment", "?")
            lines.append(
                f"• [{product}] Rating: {rating}/5 ({sentiment}) — "
                f'"{r.text[:150]}..."'
            )

        top = results[0]
        lines.append(
            f"\n📌 Top match (score={top.score:.3f}): {top.text[:200]}"
        )

        return "\n".join(lines)
