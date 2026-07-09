"""
semantic_relevance.py
=====================
Orchestrator for the FinSent semantic relevance layer (§ 4 of the report).

This module is the single entry-point that replaces the cosine-threshold
filter in method5_finbert_filtered.py with the full three-stage pipeline:

    Stage 1 — Bi-encoder recall (FAISS)
        all-MiniLM-L6-v2 encodes every article into a dense vector.
        FAISS IndexFlatIP retrieves the top-K=100 most similar articles
        to the ticker's reference query embedding.
        Cost: ~1 ms per query (after index is built).

    Stage 2 — Cross-encoder reranking
        ms-marco-MiniLM-L-6-v2 scores each (query, article) pair jointly.
        Applied only to the top-K candidates from Stage 1.
        Returns top-M=20 reranked articles.
        Cost: ~50-200 ms per ticker (batch of ≤100 pairs on CPU).

    Stage 3 — NER + KG entity boost
        EntityLinker scores each reranked article against the ticker's
        entity catalogue (primary names, aliases, executives, subsidiaries).
        The NER score contributes 20% to the _final_score already computed
        inside CrossEncoderReranker.rerank().
        GICS broadcast weights are returned alongside the final article set
        so the analytics layer can propagate macro-sector news.

Drop-in replacement contract
-----------------------------
    config.py and fetch_articles.py are unchanged.
    This module exposes one function:

        filter_articles_semantic(ticker_display, articles, window,
                                 top_k=100, top_m=20)
            → (filtered_articles: list[dict], metadata: dict)

    filtered_articles is a list of at most top_m article dicts, each
    enriched with:
        _bi_score      float  — cosine sim from bi-encoder
        _ce_score      float  — cross-encoder logit
        _ce_norm       float  — CE score normalised to [0,1]
        _ner_score     float  — NER entity-linking score
        _final_score   float  — fused relevance score
        _ce_rank       int    — final rank (0 = most relevant)

    metadata dict contains:
        n_raw          int    — articles before any filtering
        n_recall       int    — articles after bi-encoder recall (≤ top_k)
        n_reranked     int    — articles after cross-encoder (≤ top_m)
        fallback_used  bool   — True if bi-encoder returned 0 candidates
        mean_bi_score  float  — mean cosine sim of recall set
        mean_ce_norm   float  — mean CE norm score of final set
        mean_ner_score float  — mean NER score of final set
        mean_final     float  — mean _final_score of final set
        broadcast      dict   — {ticker: weight} for GICS sector articles

Integration with compare_results.py
-------------------------------------
    method6_semantic.py (new Config 6) uses this module and follows
    the same output schema as method5_finbert_filtered.run_config5():

        {ticker_display: {
            "score":         float | None,
            "n_raw":         int,
            "n_filtered":    int,
            "fallback_used": bool,
            "mean_sim":      float,
            # additional fields not in C5:
            "mean_final":    float,
            "mean_ner":      float,
            "broadcast":     dict,
        }}

Audit trail
-----------
Every inference enriches the article dict with all intermediate scores,
satisfying the FSRM audit-log requirement (§ 8.1) that each inference
produces a source-traceable record.  The calling code in method6_semantic.py
can persist these dicts as JSON.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda OpenMP fix
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from faiss_index          import recall_candidates, TICKER_REFERENCE_QUERIES
from cross_encoder_reranker import get_reranker
from ner_entity_linker    import get_entity_linker, ENTITY_CATALOGUE


# ---------------------------------------------------------------------------
# Constants  (tunable without touching any other file)
# ---------------------------------------------------------------------------

TOP_K_RECALL   = 100   # bi-encoder recall set size
TOP_M_RERANK   = 20    # cross-encoder precision set size


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def filter_articles_semantic(
    ticker_display: str,
    articles: List[dict],
    window: dict,
    top_k: int = TOP_K_RECALL,
    top_m: int = TOP_M_RERANK,
) -> Tuple[List[dict], dict]:
    """
    Full three-stage semantic relevance filter.

    Parameters
    ----------
    ticker_display : str   e.g. "JPM"
    articles       : list[dict]  normalised article dicts from fetch_articles()
    window         : dict with keys 'label', 'start', 'end'
    top_k          : int   bi-encoder recall size (default 100)
    top_m          : int   cross-encoder output size (default 20)

    Returns
    -------
    (filtered_articles, metadata)
    """
    n_raw = len(articles)
    t0    = time.time()

    # ── Stage 1: Bi-encoder recall ───────────────────────────────────────
    recall_set = recall_candidates(ticker_display, window, articles, top_k=top_k)
    n_recall   = len(recall_set)

    fallback_used = False
    if n_recall == 0:
        # No candidates survived recall — fall back to raw list (unscored)
        print(f"  [{ticker_display}] bi-encoder recall empty — using fallback")
        fallback_used = True
        # Add dummy scores so downstream code doesn't break
        for art in articles:
            art.setdefault("_bi_score",    0.0)
            art.setdefault("_bi_rank",     -1)
            art.setdefault("_ce_score",    0.0)
            art.setdefault("_ce_norm",     0.0)
            art.setdefault("_ner_score",   0.0)
            art.setdefault("_final_score", 0.0)
            art.setdefault("_ce_rank",     -1)
        metadata = _build_metadata(
            n_raw, 0, 0, fallback_used=True, articles_used=articles
        )
        return articles, metadata

    mean_bi = sum(a["_bi_score"] for a in recall_set) / n_recall

    # ── Stage 2: Cross-encoder reranking ────────────────────────────────
    reranker    = get_reranker()
    reranked    = reranker.rerank(ticker_display, recall_set, top_m=top_m)
    n_reranked  = len(reranked)

    # ── Stage 3: NER broadcast (score already embedded via _ner_score) ──
    # Collect broadcast weights from the top article's NER result
    # (the highest-ranked article is most likely to drive the alert)
    linker    = get_entity_linker()
    broadcast: Dict[str, float] = {}
    if reranked:
        top_art = reranked[0]
        ner_result = linker.score_article(
            ticker_display,
            top_art.get("title", ""),
            top_art.get("content", ""),
        )
        broadcast = ner_result.broadcast

    elapsed = round(time.time() - t0, 2)

    # Summary log line
    mean_final = (sum(a["_final_score"] for a in reranked) / n_reranked
                  if n_reranked else 0.0)
    mean_ner   = (sum(a["_ner_score"]   for a in reranked) / n_reranked
                  if n_reranked else 0.0)
    mean_ce    = (sum(a["_ce_norm"]     for a in reranked) / n_reranked
                  if n_reranked else 0.0)

    print(
        f"  {ticker_display:<12} "
        f"raw={n_raw:>3}  recall={n_recall:>3}  reranked={n_reranked:>2}  "
        f"bi={mean_bi:.3f}  ce={mean_ce:.3f}  ner={mean_ner:.3f}  "
        f"final={mean_final:.3f}  ({elapsed}s)"
    )

    metadata = {
        "n_raw":          n_raw,
        "n_recall":       n_recall,
        "n_reranked":     n_reranked,
        "fallback_used":  fallback_used,
        "mean_bi_score":  round(mean_bi,    4),
        "mean_ce_norm":   round(mean_ce,    4),
        "mean_ner_score": round(mean_ner,   4),
        "mean_final":     round(mean_final, 4),
        "broadcast":      broadcast,
    }
    return reranked, metadata


def _build_metadata(
    n_raw: int,
    n_recall: int,
    n_reranked: int,
    fallback_used: bool,
    articles_used: List[dict],
) -> dict:
    mean_f = (sum(a.get("_final_score", 0.0) for a in articles_used)
              / len(articles_used) if articles_used else 0.0)
    return {
        "n_raw":          n_raw,
        "n_recall":       n_recall,
        "n_reranked":     n_reranked,
        "fallback_used":  fallback_used,
        "mean_bi_score":  0.0,
        "mean_ce_norm":   0.0,
        "mean_ner_score": 0.0,
        "mean_final":     round(mean_f, 4),
        "broadcast":      {},
    }


# ---------------------------------------------------------------------------
# GICS broadcast helper (for analytics layer)
# ---------------------------------------------------------------------------

def apply_gics_broadcast(
    per_ticker_metadata: Dict[str, dict],
    all_ticker_scores: Dict[str, Optional[float]],
) -> Dict[str, float]:
    """
    Given per-ticker metadata (which contains .broadcast dicts from NER),
    propagate sector-level articles to same-GICS tickers.

    Returns an additive adjustment dict {ticker: float} that the analytics
    layer can add to the base sentiment aggregation.

    Logic
    -----
    For each (primary_ticker, broadcast_dict) pair:
      If primary_ticker has a non-None sentiment score:
        For each (secondary_ticker, weight) in broadcast_dict:
          adjustment[secondary_ticker] += primary_score * weight

    The weight is already bounded at 0.30 of the primary relevance score
    (set in EntityLinker.score_article).

    This is intentionally additive and small — it nudges, not overrides.
    """
    adjustments: Dict[str, float] = {}

    for primary, meta in per_ticker_metadata.items():
        primary_score = all_ticker_scores.get(primary)
        if primary_score is None:
            continue
        for secondary, w in meta.get("broadcast", {}).items():
            if secondary not in adjustments:
                adjustments[secondary] = 0.0
            adjustments[secondary] += primary_score * w

    return {k: round(v, 4) for k, v in adjustments.items()}


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Synthetic article pool — large enough to demonstrate recall + reranking
    fake_articles = [
        {"title": "JPMorgan Chase raises full-year NII guidance to $94bn",
         "content": "CEO Jamie Dimon told investors on the earnings call that net interest income "
                    "is expected to beat earlier forecasts driven by loan growth in Chase consumer.",
         "published": "2026-04-07", "source": "Reuters", "url": ""},

        {"title": "Federal Reserve signals two more rate cuts in 2026",
         "content": "The Fed's dot plot shifted dovish; JPMorgan and Goldman Sachs both rallied.",
         "published": "2026-04-06", "source": "Bloomberg", "url": ""},

        {"title": "JPM credit card delinquencies tick up in Q1 2026",
         "content": "JPMorgan's consumer credit division reported a 15bp rise in 30-day "
                    "delinquencies.  CFO Jeremy Barnum called it 'within normal range'.",
         "published": "2026-04-04", "source": "WSJ", "url": ""},

        {"title": "Goldman Sachs Q1 trading revenue hits record",
         "content": "GS FICC desk drove record quarterly revenue.  Rival JPMorgan underperformed "
                    "on fixed income but beat on investment banking.",
         "published": "2026-04-05", "source": "FT", "url": ""},

        {"title": "Basel III capital rules finalised — US banks face 9% CET1 floor",
         "content": "JPMorgan Chase, Bank of America, and Wells Fargo will need to hold more "
                    "capital under the finalised DFAST rules.  JPM's CET1 ratio stands at 15.2%.",
         "published": "2026-04-03", "source": "Reuters", "url": ""},

        {"title": "HDFC Bank Q4 NIM at record 4.2%, NPA steady",
         "content": "HDFC Bank reported a record net interest margin and stable gross NPA ratio.  "
                    "The RBI kept rates on hold, supporting credit growth.",
         "published": "2026-04-08", "source": "ET", "url": ""},

        {"title": "Broadcom AVGO beats Q2 earnings, AI revenue triples",
         "content": "Hock Tan said Broadcom's custom ASIC business from hyperscaler customers "
                    "drove AI revenue to $4.2bn.  VMware integration is on schedule.",
         "published": "2026-04-09", "source": "Barron's", "url": ""},

        {"title": "Toyota Motor cuts EV output target for fiscal 2026",
         "content": "Toyota said supply constraints at Aisin affected bZ4X production.  "
                    "The automaker reaffirmed its hybrid vehicle targets.",
         "published": "2026-04-05", "source": "Nikkei", "url": ""},

        {"title": "Chase Sapphire Reserve adds new travel benefit",
         "content": "JPMorgan's Chase Sapphire Reserve card now offers lounge access upgrades.  "
                    "The move targets high-spend customers in the premium credit card segment.",
         "published": "2026-04-06", "source": "Forbes", "url": ""},

        {"title": "Sony PlayStation 5 cumulative sales hit 52 million units",
         "content": "Sony Group's gaming segment reported record operating income.  "
                    "CEO Kenichiro Yoshida said the PS5 cycle still has two years to run.",
         "published": "2026-04-07", "source": "Nikkei", "url": ""},
    ]

    window = {"label": "3–10 Apr 2026", "start": "2026-04-03", "end": "2026-04-10"}

    print("\n" + "="*70)
    print("  semantic_relevance.py — smoke test")
    print("="*70)

    for ticker in ["JPM", "AVGO", "Toyota"]:
        print(f"\n[{ticker}]")
        filtered, meta = filter_articles_semantic(ticker, fake_articles, window,
                                                   top_k=10, top_m=5)
        print(f"  metadata: {meta}")
        print(f"  top article: {filtered[0]['title'][:70] if filtered else 'none'}")