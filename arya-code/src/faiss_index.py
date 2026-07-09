"""
faiss_index.py
==============
FAISS index builder and query module for the FinSent bi-encoder recall stage.

Role in the pipeline
---------------------
                    ┌─────────────────────────────────────────┐
   Article corpus → │  bi-encoder (all-MiniLM-L6-v2)         │
                    │  → float32 embeddings, shape (N, 384)   │
                    │  → stored in FAISS IndexFlatIP           │
                    └────────────────┬────────────────────────┘
                                     │  query: ticker reference embedding
                                     ▼
                              top-K=100 candidates
                                     │
                            [cross-encoder reranker]
                                     │
                              top-M=20 final articles

This module owns steps 1–2:
  - Build (or load) a FAISS index from a list of article dicts.
  - Query the index to return the top-K most similar articles for a ticker.

Index type: IndexFlatIP (exact inner product on L2-normalised vectors
= cosine similarity).  Appropriate for article corpus sizes up to ~100k;
upgrade to IndexIVFFlat for larger corpora.

Persistence
-----------
Indexes are serialised to disk under the same cache directory as the JSON
article cache, named:

    cache/<ticker>__<window_label_slug>__faiss.index   — FAISS binary
    cache/<ticker>__<window_label_slug>__faiss_ids.json — article id → position map

The index is rebuilt if:
  - The binary file does not exist.
  - The article list has changed (detected by comparing article count in the
    .json sidecar against len(articles)).

Usage
-----
    from faiss_index import FaissIndex

    fi = FaissIndex(ticker_display="JPM", window_label="3–10 Apr 2026")
    fi.build(articles)                # build once per (ticker, window)
    hits = fi.query(top_k=100)        # returns list[dict] ranked by cosine sim
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
# Allow running standalone without a config on the Python path
try:
    import config
    CACHE_DIR         = config.CACHE_DIR
    ST_MODEL_NAME     = config.SENTENCE_TRANSFORMER_MODEL
    REFERENCE_QUERIES = None   # imported from method5 when needed
except ImportError:
    CACHE_DIR     = os.path.join(os.path.dirname(__file__), "cache")
    ST_MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sentence-transformer singleton  (shared with method5_finbert_filtered)
# ---------------------------------------------------------------------------

_st_model: Optional[SentenceTransformer] = None

def get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        print(f"[FaissIndex] Loading sentence-transformer ({ST_MODEL_NAME}) …")
        _st_model = SentenceTransformer(ST_MODEL_NAME)
        print("[FaissIndex] Model ready.")
    return _st_model


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)


def _index_path(ticker: str, window_label: str) -> str:
    stem = f"{_slug(ticker)}__{_slug(window_label)}__faiss"
    return os.path.join(CACHE_DIR, stem + ".index")


def _ids_path(ticker: str, window_label: str) -> str:
    stem = f"{_slug(ticker)}__{_slug(window_label)}__faiss"
    return os.path.join(CACHE_DIR, stem + "_ids.json")


# ---------------------------------------------------------------------------
# Ticker reference queries
# (keep in sync with method5_finbert_filtered.REFERENCE_QUERIES)
# ---------------------------------------------------------------------------

TICKER_REFERENCE_QUERIES: dict[str, str] = {
    "JPM":      "JPMorgan Chase banking financial services earnings revenue credit",
    "AVGO":     "Broadcom semiconductor chips networking AI infrastructure earnings",
    "TSLA":     "Tesla electric vehicle EV battery autonomous driving earnings revenue",
    "HDFCBANK": "HDFC Bank India retail banking loans NPA earnings RBI",
    "RELIANCE": "Reliance Industries India petrochemical refinery telecom retail Jio",
    "TCS":      "Tata Consultancy Services IT services outsourcing software earnings",
    "INFY":     "Infosys IT services outsourcing software India earnings deal wins",
    "Toyota":   "Toyota Motor automobile car sales production EV hybrid Japan",
    "Sony":     "Sony Group electronics gaming PlayStation music entertainment Japan",
    "SoftBank": "SoftBank Group investment ARM Vision Fund technology Japan",
}


# ---------------------------------------------------------------------------
# FaissIndex class
# ---------------------------------------------------------------------------

class FaissIndex:
    """
    Wraps a FAISS IndexFlatIP for a single (ticker, window) combination.

    Attributes
    ----------
    ticker_display : str
        Display name (e.g. "JPM") used as the cache key.
    window_label : str
        Human-readable window label (e.g. "3–10 Apr 2026").
    _index : faiss.IndexFlatIP | None
        Loaded or built index.
    _articles : list[dict] | None
        The article list the index was built from, kept in memory for retrieval.
    _query_emb : np.ndarray | None
        The ticker's reference query embedding (shape (1, d)), cached after
        first computation.
    """

    def __init__(self, ticker_display: str, window_label: str):
        self.ticker_display = ticker_display
        self.window_label   = window_label
        self._index:    Optional[faiss.IndexFlatIP] = None
        self._articles: Optional[List[dict]]        = None
        self._query_emb: Optional[np.ndarray]       = None

    # ------------------------------------------------------------------ build

    def build(self, articles: List[dict], force_rebuild: bool = False) -> None:
        """
        Encode articles and build the FAISS index.
        Loads from disk if a valid cached index exists, unless force_rebuild.

        Parameters
        ----------
        articles : list of normalised article dicts (title, content, …)
        force_rebuild : skip disk cache and always rebuild
        """
        ipath = _index_path(self.ticker_display, self.window_label)
        jpath = _ids_path(self.ticker_display, self.window_label)

        # Attempt to load from disk
        if not force_rebuild and os.path.exists(ipath) and os.path.exists(jpath):
            try:
                with open(jpath, "r") as f:
                    meta = json.load(f)
                if meta.get("n_articles") == len(articles):
                    self._index    = faiss.read_index(ipath)
                    self._articles = articles
                    print(f"  [FaissIndex] Loaded cached index for "
                          f"{self.ticker_display} | {self.window_label} "
                          f"({len(articles)} articles)")
                    return
            except Exception as exc:
                print(f"  [FaissIndex] Cache load failed ({exc}), rebuilding …")

        if not articles:
            print(f"  [FaissIndex] No articles for {self.ticker_display} — skipping build")
            self._articles = []
            return

        # Encode
        st = get_st_model()
        char_limit = 512
        texts = [
            f"{a.get('title', '')}. {a.get('content', '')}"[:char_limit]
            for a in articles
        ]
        print(f"  [FaissIndex] Encoding {len(texts)} articles for "
              f"{self.ticker_display} | {self.window_label} …")
        embs = st.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-norm → inner product = cosine
            batch_size=32,
            show_progress_bar=False,
        ).astype(np.float32)

        # Build index
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)   # inner product on normalised = cosine
        index.add(embs)

        # Save
        faiss.write_index(index, ipath)
        with open(jpath, "w") as f:
            json.dump({"n_articles": len(articles)}, f)

        self._index    = index
        self._articles = articles
        print(f"  [FaissIndex] Built index: {len(articles)} vectors, d={d}")

    # ------------------------------------------------------------------ query

    def _get_query_embedding(self) -> np.ndarray:
        """Return cached ticker reference embedding, shape (1, d)."""
        if self._query_emb is not None:
            return self._query_emb
        st  = get_st_model()
        q   = TICKER_REFERENCE_QUERIES.get(self.ticker_display, self.ticker_display)
        emb = st.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        self._query_emb = emb
        return emb

    def query(self, top_k: int = 100) -> List[dict]:
        """
        Return the top-K most similar articles to the ticker reference query.

        Each returned dict is the original article dict enriched with:
            _bi_score  : float — cosine similarity from FAISS
            _bi_rank   : int   — rank (0 = most similar)

        Returns [] if the index was never built or has no articles.
        """
        if self._index is None or not self._articles:
            return []

        k = min(top_k, len(self._articles))
        query_emb = self._get_query_embedding()

        scores, indices = self._index.search(query_emb, k)
        scores  = scores[0].tolist()    # shape (k,)
        indices = indices[0].tolist()

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:   # FAISS returns -1 for padding when k > n
                continue
            art = dict(self._articles[idx])   # shallow copy
            art["_bi_score"] = round(float(score), 4)
            art["_bi_rank"]  = rank
            results.append(art)

        return results

    # ------------------------------------------------------------------ utils

    @property
    def n_articles(self) -> int:
        return len(self._articles) if self._articles else 0

    def clear_cache(self) -> None:
        """Delete the on-disk index files for this (ticker, window)."""
        for path in [_index_path(self.ticker_display, self.window_label),
                     _ids_path(self.ticker_display,   self.window_label)]:
            if os.path.exists(path):
                os.remove(path)
                print(f"  [FaissIndex] Deleted {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Convenience: build + query in one call (used by semantic_relevance.py)
# ---------------------------------------------------------------------------

def recall_candidates(
    ticker_display: str,
    window: dict,
    articles: List[dict],
    top_k: int = 100,
) -> List[dict]:
    """
    Build the FAISS index for (ticker, window) and return top-K candidates.
    Caches the index to disk for reuse across runs.

    Parameters
    ----------
    ticker_display : str
    window : dict with keys 'label', 'start', 'end'
    articles : full article list from fetch_articles()
    top_k : int — number of candidates to pass to the cross-encoder stage

    Returns
    -------
    list[dict] — up to top_k article dicts, sorted by cosine similarity desc.
                 Each dict has _bi_score and _bi_rank added.
    """
    fi = FaissIndex(ticker_display, window["label"])
    fi.build(articles)
    return fi.query(top_k=top_k)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic article list for testing without API access
    fake_articles = [
        {"title": "JPMorgan Chase raises net interest income forecast",
         "content": "CEO Jamie Dimon said the bank expects NII to exceed $90bn this year.",
         "published": "2026-04-07", "source": "Reuters", "url": ""},
        {"title": "Federal Reserve holds rates steady",
         "content": "The Fed kept its benchmark rate unchanged at 5.25-5.5%.",
         "published": "2026-04-06", "source": "Bloomberg", "url": ""},
        {"title": "Toyota halts production at Aichi plant",
         "content": "Supply disruption at Aisin affects Toyota Motor Q1 output.",
         "published": "2026-04-05", "source": "Nikkei", "url": ""},
        {"title": "JPM credit card delinquencies rise in Q1",
         "content": "JPMorgan's consumer credit book shows early stress signals.",
         "published": "2026-04-04", "source": "WSJ", "url": ""},
        {"title": "HDFC Bank Q4 results beat estimates",
         "content": "HDFC Bank reports record NIM of 4.2% and low NPA.",
         "published": "2026-04-08", "source": "ET", "url": ""},
    ]

    window = {"label": "3–10 Apr 2026", "start": "2026-04-03", "end": "2026-04-10"}
    hits = recall_candidates("JPM", window, fake_articles, top_k=3)

    print("\n=== FaissIndex smoke test (JPM, top-3) ===")
    for h in hits:
        print(f"  rank={h['_bi_rank']}  score={h['_bi_score']:.4f}  "
              f"title={h['title'][:60]}")