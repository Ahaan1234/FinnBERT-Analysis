"""
cross_encoder_reranker.py
=========================
Cross-encoder reranking (precision stage) for the FinSent semantic relevance layer.

Role in the pipeline
---------------------
                    ┌──────────────────────────────────────────┐
  top-K=100 cands  │  cross-encoder (MiniLM-L-6-v2)           │
  from bi-encoder  │  scores each (query, article) pair jointly │
                   │  → relevance logit / probability            │
                   └────────────────┬─────────────────────────-┘
                                    │
                             top-M=20 reranked
                                    │
                        [NER entity-level boost]
                                    │
                            final ranked set
                            → FinBERT scoring

Why cross-encoders outperform bi-encoders for precision
-------------------------------------------------------
A bi-encoder encodes query and document independently; the interaction
between them is limited to a dot product in the embedding space.  A
cross-encoder takes the (query, document) pair as a single input, so
the transformer's attention layers can model fine-grained relevance
patterns — e.g. "article mentions JPMorgan only to contrast with Goldman"
gets a low score because the cross-attention sees the full context.

Cost trade-off: cross-encoders are ~50-100× slower per pair, which is why
we run them only on the top-K=100 recall candidates, not the full corpus.

Model used
----------
  sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2

This is a MSMARCO-trained passage-reranking model: given a (query, passage)
pair it produces a relevance score (logit, not probability).  It is small
(~66M params), fast enough for batched reranking of 100 candidates on CPU,
and gives strong precision improvements over cosine-only approaches.

The query passed to the cross-encoder is the same ticker reference query
used by the bi-encoder (kept in TICKER_REFERENCE_QUERIES).

Usage
-----
    from cross_encoder_reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(
        ticker_display = "JPM",
        candidates     = bi_encoder_hits,   # list[dict] from faiss_index.query()
        top_m          = 20,
    )
    # reranked is a list[dict] sorted by cross-encoder score desc.
    # Each dict has:
    #   _bi_score    : float  (from bi-encoder)
    #   _ce_score    : float  (cross-encoder logit, higher = more relevant)
    #   _ce_rank     : int    (rank after reranking; 0 = most relevant)
    #   _ner_score   : float  (NER entity-linking score, added here)
    #   _final_score : float  (combined score used downstream)

Score fusion
------------
final_score = 0.5 * norm(ce_score) + 0.3 * bi_score + 0.2 * ner_score

  norm(ce_score): cross-encoder logit normalised to [0,1] via min-max
                  across the candidate set (done per-batch).
  bi_score:       already in [0,1] from cosine similarity.
  ner_score:      EntityLinker.score_article().score  ∈ [0,1].

Weights (0.5 / 0.3 / 0.2) are a reasonable default backed by the observation
in results.txt that cosine-only and composite-metric agree on only 30% of
top-20 articles — the cross-encoder provides the tie-breaking precision.

All weights are configurable via constructor args for future tuning.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

# Suppress tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder

try:
    from ner_entity_linker import get_entity_linker
    _NER_AVAILABLE = True
except ImportError:
    _NER_AVAILABLE = False

try:
    from faiss_index import TICKER_REFERENCE_QUERIES
except ImportError:
    TICKER_REFERENCE_QUERIES = {}


# ---------------------------------------------------------------------------
# Model choice
# ---------------------------------------------------------------------------

CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# CrossEncoderReranker class
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Reranks bi-encoder candidates using a cross-encoder, then fuses with
    NER entity-linking scores.

    Parameters
    ----------
    ce_model_name : str
        HuggingFace model id for the cross-encoder.
    w_ce : float
        Weight of normalised cross-encoder score in final fusion (default 0.5).
    w_bi : float
        Weight of bi-encoder cosine score (default 0.3).
    w_ner : float
        Weight of NER entity-linking score (default 0.2).
    char_limit : int
        Max chars of article text passed to the cross-encoder (default 512).
    """

    def __init__(
        self,
        ce_model_name: str = CE_MODEL_NAME,
        w_ce:   float = 0.50,
        w_bi:   float = 0.30,
        w_ner:  float = 0.20,
        char_limit: int = 512,
    ):
        self.w_ce       = w_ce
        self.w_bi       = w_bi
        self.w_ner      = w_ner
        self.char_limit = char_limit

        print(f"[CrossEncoder] Loading {ce_model_name} …")
        self._model = CrossEncoder(ce_model_name, max_length=512)
        print("[CrossEncoder] Model ready.")

        self._linker = get_entity_linker() if _NER_AVAILABLE else None

    # ------------------------------------------------------------------

    def rerank(
        self,
        ticker_display: str,
        candidates: List[dict],
        top_m: int = 20,
    ) -> List[dict]:
        """
        Rerank bi-encoder candidates and return the top-M.

        Parameters
        ----------
        ticker_display : str
            Ticker key (e.g. "JPM").
        candidates : list[dict]
            Output of faiss_index.query() — article dicts with _bi_score.
        top_m : int
            How many to return after reranking.

        Returns
        -------
        list[dict] sorted by _final_score descending, length <= top_m.
        Each dict has _ce_score, _ce_rank, _ner_score, _final_score added.
        """
        if not candidates:
            return []

        query = TICKER_REFERENCE_QUERIES.get(ticker_display, ticker_display)
        cl    = self.char_limit

        # Build (query, passage) pairs for the cross-encoder
        pairs = []
        for art in candidates:
            passage = f"{art.get('title', '')}. {art.get('content', '')}"[:cl]
            pairs.append([query, passage])

        # Score all pairs in one batched call
        ce_scores: np.ndarray = self._model.predict(pairs, show_progress_bar=False)
        # ce_scores shape: (n_candidates,), logit scale — higher = more relevant

        # Normalise CE scores to [0, 1] via min-max across this candidate set
        ce_min, ce_max = float(ce_scores.min()), float(ce_scores.max())
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        ce_norm  = ((ce_scores - ce_min) / ce_range).tolist()

        # NER scores
        ner_scores: List[float] = []
        for art in candidates:
            if self._linker is not None:
                r = self._linker.score_article(
                    ticker_display,
                    art.get("title", ""),
                    art.get("content", ""),
                )
                ner_scores.append(r.score)
            else:
                ner_scores.append(0.0)

        # Fuse scores and annotate each article dict
        fused = []
        for i, (art, ce_raw, ce_n, ner) in enumerate(
            zip(candidates, ce_scores.tolist(), ce_norm, ner_scores)
        ):
            bi = art.get("_bi_score", 0.0)
            final = self.w_ce * ce_n + self.w_bi * bi + self.w_ner * ner
            enriched = dict(art)   # shallow copy
            enriched["_ce_score"]    = round(float(ce_raw), 4)
            enriched["_ce_norm"]     = round(float(ce_n),   4)
            enriched["_ner_score"]   = round(float(ner),    4)
            enriched["_final_score"] = round(float(final),  4)
            fused.append(enriched)

        # Sort by final score descending
        fused.sort(key=lambda x: x["_final_score"], reverse=True)

        # Assign CE rank
        for rank, art in enumerate(fused):
            art["_ce_rank"] = rank

        return fused[:top_m]

    # ------------------------------------------------------------------

    def explain(self, article: dict) -> str:
        """
        Human-readable score breakdown for an article dict that has been
        through rerank().  Used by the explainability / audit layer.
        """
        bi    = article.get("_bi_score",    "n/a")
        ce    = article.get("_ce_norm",     "n/a")
        ner   = article.get("_ner_score",   "n/a")
        final = article.get("_final_score", "n/a")
        title = article.get("title", "")[:80]
        return (
            f'"{title}"\n'
            f"  bi_score={bi}  ce_norm={ce}  ner_score={ner}  "
            f"→ final={final}  "
            f"(w_ce={self.w_ce}, w_bi={self.w_bi}, w_ner={self.w_ner})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_reranker_singleton: Optional[CrossEncoderReranker] = None

def get_reranker() -> CrossEncoderReranker:
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = CrossEncoderReranker()
    return _reranker_singleton


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic candidates mimicking faiss_index output
    fake_candidates = [
        {"title": "JPMorgan Chase raises net interest income forecast",
         "content": "CEO Jamie Dimon said NII will exceed $90bn.",
         "published": "2026-04-07", "source": "Reuters", "url": "",
         "_bi_score": 0.82, "_bi_rank": 0},
        {"title": "Federal Reserve holds rates steady in April",
         "content": "The Fed kept its benchmark rate unchanged.",
         "published": "2026-04-06", "source": "Bloomberg", "url": "",
         "_bi_score": 0.61, "_bi_rank": 1},
        {"title": "JPM credit card delinquencies rise Q1",
         "content": "JPMorgan consumer credit shows early stress signals.",
         "published": "2026-04-04", "source": "WSJ", "url": "",
         "_bi_score": 0.74, "_bi_rank": 2},
        {"title": "Goldman Sachs beats estimates on trading revenue",
         "content": "GS reported record FICC revenue for the quarter.",
         "published": "2026-04-05", "source": "FT", "url": "",
         "_bi_score": 0.45, "_bi_rank": 3},
        {"title": "HDFC Bank Q4 NIM reaches record high",
         "content": "HDFC Bank reports 4.2% NIM; CASA ratio improves.",
         "published": "2026-04-08", "source": "ET", "url": "",
         "_bi_score": 0.30, "_bi_rank": 4},
    ]

    reranker = CrossEncoderReranker()
    results  = reranker.rerank("JPM", fake_candidates, top_m=3)

    print("\n=== CrossEncoderReranker smoke test (JPM, top-3) ===")
    for r in results:
        print(reranker.explain(r))