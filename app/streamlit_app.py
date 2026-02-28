"""
Streamlit Demo — Semantic Search & RAG on Flipkart Reviews

Three pages:
  1. Search — Semantic + Hybrid search with product filter
  2. RAG Chat — Ask questions, get grounded answers
  3. Embedding Viz — Interactive PCA/UMAP scatter

Launch:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('SAMPLE_ONLY', 'true')

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Flipkart Semantic Search", page_icon="🔍", layout="wide")


@st.cache_resource
def load_system():
    """Load all components once."""
    from src.config import Config
    from src.data_ingest import load_flipkart, get_product_names
    from src.embedding_model import EmbeddingModel
    from src.indexer import FAISSIndexer
    from src.retriever import DenseRetriever
    from src.hybrid_search import HybridSearcher
    from src.reranker import Reranker
    from src.rag_pipeline import RAGPipeline
    from src.utils import load_pickle

    cfg = Config()
    df = load_flipkart(cfg)
    texts = df['combined_text'].tolist()
    metadata = df.to_dict('records')

    emb = EmbeddingModel(cfg)
    try:
        vectors = load_pickle(cfg.DATA_PROCESSED / 'embeddings.pkl')
    except FileNotFoundError:
        vectors = emb.encode(texts, normalize=True)

    indexer = FAISSIndexer(dim=emb.dim, index_type='hnsw', cfg=cfg)
    indexer.add(vectors)

    retriever = DenseRetriever(indexer, emb, texts, metadata)
    hybrid = HybridSearcher(retriever, texts, metadata, alpha=0.6)
    reranker = Reranker()
    rag = RAGPipeline(retriever, reranker, top_k=10, rerank_k=5)
    products = get_product_names(df, cfg)

    return retriever, hybrid, rag, products, vectors, texts, metadata, df, cfg


def main():
    retriever, hybrid, rag, products, vectors, texts, metadata, df, cfg = load_system()

    page = st.sidebar.radio("📄 Page", ["🔍 Search", "🤖 RAG Chat", "📊 Embedding Viz"])

    if page == "🔍 Search":
        st.title("🔍 Semantic Product Search")
        query = st.text_input("Search query", "good battery life")
        product_filter = st.selectbox("Product filter", ["All"] + products)
        search_type = st.radio("Search type", ["Semantic", "Hybrid"], horizontal=True)
        k = st.slider("Results", 3, 15, 5)

        if st.button("Search") and query:
            pf = None if product_filter == "All" else product_filter[:20]
            if search_type == "Semantic":
                results = retriever.query(query, k=k, product_filter=pf)
            else:
                results = hybrid.query(query, k=k, product_filter=pf)

            for r in results:
                product = str(r.metadata.get('product_name', ''))[:40]
                rating = r.metadata.get('Rating', '?')
                sentiment = r.metadata.get('Sentiment', '?')
                with st.expander(f"#{r.rank} — {product} ⭐{rating} [{sentiment}] (score: {r.score:.3f})"):
                    st.write(r.text)

    elif page == "🤖 RAG Chat":
        st.title("🤖 RAG — Ask About Products")
        query = st.text_input("Ask a question", "Which product is best for summer cooling?")
        product_filter = st.selectbox("Limit to product", ["All"] + products)

        if st.button("Ask") and query:
            pf = None if product_filter == "All" else product_filter[:20]
            answer = rag.answer(query, product_filter=pf)
            st.markdown("### Answer")
            st.write(answer)

    elif page == "📊 Embedding Viz":
        st.title("📊 Embedding Visualization")
        from src.visualization import plot_embeddings_2d
        color_by = st.selectbox("Color by", ["Sentiment", "Product"])

        if color_by == "Sentiment":
            labels = df[cfg.COL_SENTIMENT].values
        else:
            labels = [n[:25] for n in df[cfg.COL_PRODUCT].values]

        fig = plot_embeddings_2d(vectors, labels=labels, method='pca',
                                title=f'PCA — Colored by {color_by}')
        st.pyplot(fig)


if __name__ == "__main__":
    main()
