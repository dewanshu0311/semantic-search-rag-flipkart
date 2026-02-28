"""
Manual Evaluation Labelling App.

Label retrieved documents as relevant/not relevant to build ground truth
for Precision@K and Recall@K computation.

Run: ``streamlit run app/manual_eval_app.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Manual Eval UI", page_icon="📝", layout="wide")
st.title("📝 Manual Relevance Labelling")

LABELS_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "manual_labels.csv"

# Load existing labels
if LABELS_PATH.exists():
    existing = pd.read_csv(LABELS_PATH)
else:
    existing = pd.DataFrame(columns=["query", "doc_id", "text_preview", "relevant"])

st.sidebar.metric("Labels Collected", len(existing))

query = st.text_input("Search Query:", placeholder="e.g. best protein bar")
k = st.slider("Top-K to review", 3, 10, 5)

if query:

    @st.cache_resource
    def get_retriever():
        from src.config import Config
        from src.data_ingest import load_reviews
        from src.embedding_model import EmbeddingModel
        from src.indexer import FAISSIndexer
        from src.retriever import DenseRetriever

        cfg = Config()
        cfg.SAMPLE_ONLY = True
        cfg.SAMPLE_SIZE = 500
        df = load_reviews(cfg)
        texts = df["Text"].tolist()
        meta = df.to_dict("records")
        emb = EmbeddingModel(cfg)
        vecs = emb.encode(texts, show_progress=False)
        idx = FAISSIndexer(dim=emb.dim, index_type="flat")
        idx.add(vecs)
        return DenseRetriever(idx, emb, texts, meta)

    try:
        retriever = get_retriever()
    except FileNotFoundError:
        st.error("Place Reviews.csv in data/raw/ first.")
        st.stop()

    results = retriever.query(query, k=k)

    labels = {}
    for r in results:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**[Score: {r.score:.4f}]** {r.text[:300]}...")
        with col2:
            labels[r.metadata.get("Id", r.rank)] = st.selectbox(
                "Relevant?", ["Yes", "No"], key=f"label_{r.rank}"
            )

    if st.button("💾 Save Labels"):
        rows = []
        for doc_id, label in labels.items():
            rows.append({
                "query": query,
                "doc_id": doc_id,
                "text_preview": "",
                "relevant": 1 if label == "Yes" else 0,
            })
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(LABELS_PATH, index=False)
        st.success(f"Saved {len(rows)} labels → {LABELS_PATH}")
        st.rerun()
