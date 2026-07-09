"""
method6_semantic.py
===================
Config 6 — Full Semantic Relevance + FinBERT.

This is the production-tier replacement for Config 5.  It runs the complete
three-stage semantic relevance layer (bi-encoder → cross-encoder → NER/KG)
before FinBERT scoring, matching the architecture described in §4 of the
FinSent report.

Pipeline per ticker
-------------------
    fetch_articles()              [unchanged from C4/C5]
         │
         ▼
    Stage 1: FAISS bi-encoder recall (top-K=100)
    Stage 2: Cross-encoder reranking (top-M=20)
    Stage 3: NER entity boost + GICS broadcast scores
         │
         ▼
    FinBERT (batched, same as C4/C5)
    → mean(P(positive) - P(negative)) over filtered set

Output schema
-------------
Identical to run_config5() so compare_results.py can slot in C6 with one
line change.  Additional fields are passed through and printed in the
article-coverage table.

    {ticker_display: {
        "score":         float | None,
        "n_raw":         int,
        "n_filtered":    int,     # = n_reranked (articles after Stage 2)
        "fallback_used": bool,
        "mean_sim":      float,   # = mean_final (fused relevance score)
        # C6 extras:
        "n_recall":      int,     # articles after Stage 1 (≤ top_k)
        "mean_bi":       float,
        "mean_ce":       float,
        "mean_ner":      float,
        "broadcast":     dict,    # {ticker: weight} for GICS sector
    }}

Integrating into compare_results.py
-------------------------------------
1.  Add this import at the top:
        from method6_semantic import run_config6

2.  Add c6 = run_config6(window) alongside c3/c4/c5 in main().

3.  Update COL_LABELS to include "C6 Semantic+FB" and pass c6 to the table
    helpers — all formatters work on the same dict schema.

Usage
-----
    python method6_semantic.py                  # run both windows
    python method6_semantic.py --cache-only     # use cached articles + index
    python method6_semantic.py --window 0       # run only first window (0-indexed)
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Allow running from arya-code/src or from the semantic_relevance folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config
from fetch_articles          import fetch_articles
from method3_finbert_no_data import get_finbert, _scores_to_scalar
from semantic_relevance      import filter_articles_semantic, apply_gics_broadcast


# ---------------------------------------------------------------------------
# FinBERT batch scorer  (same logic as method4 / method5)
# ---------------------------------------------------------------------------

def _score_articles(articles: list[dict]) -> list[float]:
    clf        = get_finbert()
    char_limit = config.FINBERT_MAX_LENGTH * 4
    texts      = [
        f"{a['title']}. {a['content']}"[:char_limit]
        for a in articles
    ]
    scores = []
    bs = config.FINBERT_BATCH_SIZE
    for i in range(0, len(texts), bs):
        batch   = texts[i : i + bs]
        results = clf(batch, truncation=True, max_length=config.FINBERT_MAX_LENGTH)
        for result in results:
            if isinstance(result, list) and isinstance(result[0], dict):
                scores.append(_scores_to_scalar(result))
            elif isinstance(result, list) and isinstance(result[0], list):
                scores.append(_scores_to_scalar(result[0]))
            else:
                scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Config 6 runner
# ---------------------------------------------------------------------------

def run_config6(
    window: dict,
    top_k: int = 100,
    top_m: int = 20,
) -> dict[str, dict]:
    """
    Run the full semantic relevance pipeline + FinBERT for one date window.

    Parameters
    ----------
    window : dict with keys 'label', 'start', 'end'
    top_k  : bi-encoder recall size (default 100)
    top_m  : cross-encoder output size, i.e. articles passed to FinBERT (default 20)

    Returns
    -------
    dict  {ticker_display: result_dict}
    """
    print(f"\n[Config 7 — Semantic] Window: {window['label']}")
    print(f"  bi-encoder recall top-K={top_k}  |  cross-encoder top-M={top_m}")

    results: dict[str, dict] = {}
    all_scores: dict[str, float | None] = {}
    all_meta:   dict[str, dict] = {}

    for ticker_display in config.TICKER_ORDER:

        # ── 1. Fetch articles (cached if available) ──────────────────────
        articles = fetch_articles(ticker_display, window)
        n_raw    = len(articles)

        if n_raw == 0:
            print(f"  {ticker_display:<12} NO ARTICLES — score=None")
            results[ticker_display] = {
                "score": None, "n_raw": 0, "n_filtered": 0,
                "n_recall": 0, "fallback_used": False,
                "mean_sim": 0.0, "mean_bi": 0.0,
                "mean_ce": 0.0, "mean_ner": 0.0,
                "broadcast": {},
            }
            all_scores[ticker_display] = None
            all_meta[ticker_display]   = {"broadcast": {}}
            continue

        # ── 2. Semantic relevance filter ─────────────────────────────────
        filtered, meta = filter_articles_semantic(
            ticker_display, articles, window,
            top_k=top_k, top_m=top_m,
        )
        n_filtered    = len(filtered)
        fallback_used = meta["fallback_used"]

        if n_filtered == 0:
            print(f"  {ticker_display:<12} semantic filter returned 0 — "
                  f"score=None")
            results[ticker_display] = {
                "score": None, "n_raw": n_raw, "n_filtered": 0,
                "n_recall": meta["n_recall"], "fallback_used": True,
                "mean_sim": 0.0, "mean_bi": meta["mean_bi_score"],
                "mean_ce": meta["mean_ce_norm"], "mean_ner": meta["mean_ner_score"],
                "broadcast": meta.get("broadcast", {}),
            }
            all_scores[ticker_display] = None
            all_meta[ticker_display]   = meta
            continue

        # ── 3. FinBERT scoring ───────────────────────────────────────────
        per_article_scores = _score_articles(filtered)
        mean_score = sum(per_article_scores) / len(per_article_scores)

        fb_marker = " [FALLBACK]" if fallback_used else ""
        print(
            f"  {ticker_display:<12} "
            f"finbert_score={mean_score:+.4f}  "
            f"n_finbert={n_filtered}{fb_marker}"
        )

        results[ticker_display] = {
            "score":         round(mean_score, 4),
            "n_raw":         n_raw,
            "n_filtered":    n_filtered,
            "n_recall":      meta["n_recall"],
            "fallback_used": fallback_used,
            "mean_sim":      meta["mean_final"],       # fused score — analogous to mean_sim in C5
            "mean_bi":       meta["mean_bi_score"],
            "mean_ce":       meta["mean_ce_norm"],
            "mean_ner":      meta["mean_ner_score"],
            "broadcast":     meta.get("broadcast", {}),
        }
        all_scores[ticker_display] = round(mean_score, 4)
        all_meta[ticker_display]   = meta

    # ── 4. GICS broadcast adjustments ────────────────────────────────────
    broadcast_adj = apply_gics_broadcast(all_meta, all_scores)
    if broadcast_adj:
        print(f"\n  [Config 6] GICS broadcast adjustments: {broadcast_adj}")

    # Attach broadcast_adj to each result for the analytics layer
    for ticker, adj in broadcast_adj.items():
        if ticker in results and results[ticker]["score"] is not None:
            results[ticker]["gics_broadcast_adj"] = adj

    return results


# ---------------------------------------------------------------------------
# Pretty-print helper  (mirrors compare_results.py's print_article_coverage)
# ---------------------------------------------------------------------------

def print_config6_table(window_label: str, c6: dict[str, dict]) -> None:
    """
    Print a detailed per-ticker table for Config 6 results.
    Shows all three relevance scores + FinBERT score.
    """
    TW = 12
    sep = (
        "+" + "-"*TW +
        "+" + "-"*7  +   # n_raw
        "+" + "-"*8  +   # n_recall
        "+" + "-"*10 +   # n_reranked
        "+" + "-"*8  +   # mean_bi
        "+" + "-"*8  +   # mean_ce
        "+" + "-"*8  +   # mean_ner
        "+" + "-"*10 +   # finbert
        "+"
    )
    print(f"\n{'='*85}")
    print(f"  Config 6 — Semantic Relevance + FinBERT  |  {window_label}")
    print(f"{'='*85}")
    print(sep)
    print(
        "|" + "Ticker".center(TW) +
        "|" + "Raw".center(7) +
        "|" + "Recall".center(8) +
        "|" + "Reranked".center(10) +
        "|" + "Bi".center(8) +
        "|" + "CE".center(8) +
        "|" + "NER".center(8) +
        "|" + "FinBERT".center(10) + "|"
    )
    print(sep)

    for ticker, r in c6.items():
        score_str = f"{r['score']:+.4f}" if r["score"] is not None else "N/A"
        fb_mark   = "*" if r.get("fallback_used") else " "
        print(
            "|" + ticker.center(TW) +
            "|" + str(r["n_raw"]).center(7) +
            "|" + str(r.get("n_recall", "—")).center(8) +
            "|" + str(r["n_filtered"]).center(10) +
            "|" + f"{r.get('mean_bi', 0.0):.3f}".center(8) +
            "|" + f"{r.get('mean_ce', 0.0):.3f}".center(8) +
            "|" + f"{r.get('mean_ner', 0.0):.3f}".center(8) +
            "|" + (score_str + fb_mark).center(10) + "|"
        )

    print(sep)
    print("  * fallback: bi-encoder returned 0 candidates; unfiltered used.\n")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse optional --window N flag
    window_idx: int | None = None
    if "--window" in sys.argv:
        try:
            idx = sys.argv.index("--window")
            window_idx = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("[WARN] --window requires an integer argument; running all windows.")

    windows = config.DATE_WINDOWS
    if window_idx is not None:
        windows = [windows[window_idx]]

    print("\n" + "="*80)
    print(" Config 7(Semantic Relevance + FinBERT)")
    print("="*80)

    all_results = {}
    t_total = time.time()

    for window in windows:
        lbl = window["label"]
        t0  = time.time()
        c6  = run_config6(window)
        elapsed = time.time() - t0
        all_results[lbl] = c6
        print(f"\n  [timing] {lbl} completed in {elapsed:.1f}s")

    # Print tables
    for window in windows:
        lbl = window["label"]
        print_config6_table(lbl, all_results[lbl])

    print("\n[Done] Config 7 complete.\n")