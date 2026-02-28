# Presentation: Semantic Search & RAG System

## Slide 1 — Title
- **Semantic Search & Retrieval-Augmented Generation System**
- Built on Amazon Fine Food Reviews & Flipkart Product Reviews
- NIAT Vector Search Masterclass — Final Project

## Slide 2 — Problem Statement
- Keyword search fails to capture user intent
- Pure LLMs hallucinate without grounding in real data
- Challenge: Build a system that retrieves *and* understands

## Slide 3 — Architecture Overview
- Sentence-BERT embeddings (384-dim dense vectors)
- FAISS indexing (Flat / HNSW / IVF-PQ)
- Hybrid search: BM25 + FAISS with tunable alpha
- Cross-encoder / sentiment-aware re-ranking
- RAG generation with OpenAI or local LLM fallback

## Slide 4 — Key Results
- Sub-millisecond retrieval on 50K vectors (HNSW)
- 12-18% Precision@5 improvement with hybrid search
- Sentiment-aware re-ranker surfaces high-quality products
- RAG reduces information overload by ~80%

## Slide 5 — Index Benchmarking
- Flat: Perfect recall, O(N) speed
- HNSW: 99%+ recall, ~0.5ms search
- IVF-PQ: 10x memory reduction, 90%+ recall
- (Show benchmarking charts from Notebook 06)

## Slide 6 — Live Demo
- Streamlit search interface
- RAG chat panel
- Embedding visualization (PCA/UMAP)
- Manual evaluation labelling

## Slide 7 — Business Impact
- Customer support automation
- Product intelligence & competitive analysis
- Recommendation engine prototype
- Content generation from review aggregation

## Slide 8 — Future Work
- Query rewriting for improved recall
- Multi-modal search (images + text)
- Scaling to 1M+ vectors with distributed FAISS
- Fine-tuning embedding model on domain data
