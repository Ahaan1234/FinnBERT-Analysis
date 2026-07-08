"""
run_llm_comparison.py
=====================
FinSent benchmark — LLM vs FinBERT comparison run.

Purpose
-------
Reproduce the "EODHD + pre-filtering + FinBERT" column in the EY slide deck
table alongside the EYQ GPT 5.1 results already computed by Atharva/Dilip.

Tickers and windows (per-ticker, matching EYQ run):
    INTC  (Intel Corporation)   27 Apr – 01 May 2026
    TM    (Toyota Motors, US)   02 Mar – 06 Mar 2026
    SONY  (Sony Group, US ADR)  08 Jun – 12 Jun 2026
    DELL  (Dell Technologies)   25 May – 29 May 2026

All four are US-listed → all fetched via EODHD.
Pipeline: EODHD fetch → cosine pre-filter → FinBERT (C5 only, since C3 is
uninformative without articles and C4 is the unfiltered baseline).

Both C4 (unfiltered) and C5 (filtered) scores are printed so the delta
is visible, but the headline column for the slide deck is C5.

GPT 5.1 scores from the image are hardcoded for side-by-side comparison.

Usage
-----
    python run_llm_comparison.py              # fetch + run
    python run_llm_comparison.py --cache-only # use cached articles, no API calls
"""

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda OpenMP fix

import config  # shared config — API keys, EODHD params, model settings
from cache_utils import load_cache, save_cache
from fetch_articles import fetch_eodhd, _in_window
from method4_finbert_articles import _score_articles
from method5_finbert_filtered import filter_articles, get_st_model

CACHE_ONLY = "--cache-only" in sys.argv

# ---------------------------------------------------------------------------
# Per-ticker configuration for this run
# All US-listed → EODHD for all four
# ---------------------------------------------------------------------------

TICKERS = [
    {
        "display":  "INTC",
        "symbol":   "INTC",
        "name":     "Intel Corporation",
        "start":    "2026-04-27",
        "end":      "2026-05-01",
        "label":    "27 Apr–01 May 2026",
        "gpt_score": 0.5228,     # EYQ GPT 5.1 result from slide table
        "returns":  "+19.06%",
        "n_articles_gpt": 458,
    },
    {
        "display":  "TM",
        "symbol":   "TM",
        "name":     "Toyota Motors (US)",
        "start":    "2026-03-02",
        "end":      "2026-03-06",
        "label":    "02 Mar–06 Mar 2026",
        "gpt_score": -0.0170,
        "returns":  "-9.81%",
        "n_articles_gpt": 23,
    },
    {
        "display":  "SONY",
        "symbol":   "SONY",
        "name":     "Sony Group (US ADR)",
        "start":    "2026-06-08",
        "end":      "2026-06-12",
        "label":    "08 Jun–12 Jun 2026",
        "gpt_score": -0.1575,
        "returns":  "-7.01%",
        "n_articles_gpt": 10,
    },
    {
        "display":  "DELL",
        "symbol":   "DELL",
        "name":     "Dell Technologies",
        "start":    "2026-05-25",
        "end":      "2026-05-29",
        "label":    "25 May–29 May 2026",
        "gpt_score": 0.6671,
        "returns":  "+40.18%",
        "n_articles_gpt": 238,
    },
]

# Reference queries for cosine pre-filter — company-specific semantic anchors
REFERENCE_QUERIES = {
    "INTC": "Intel Corporation semiconductor CPU processor chips earnings revenue datacenter AI",
    "TM":   "Toyota Motor automobile car sales production EV hybrid vehicle Japan revenue",
    "SONY": "Sony Group electronics gaming PlayStation music entertainment Japan earnings",
    "DELL": "Dell Technologies PC server storage enterprise hardware earnings revenue cloud",
}


# ---------------------------------------------------------------------------
# Fetch helper — wraps fetch_eodhd but builds a window dict on the fly
# ---------------------------------------------------------------------------

def fetch_for_ticker(t: dict) -> list[dict]:
    """
    Fetch EODHD articles for a single ticker config dict.
    Respects --cache-only flag: if cache exists, use it;
    if not and --cache-only is set, return empty list with warning.
    """
    window = {
        "label": t["label"],
        "start": t["start"],
        "end":   t["end"],
    }

    # Check cache first regardless of flag
    cached = load_cache(t["display"], window["label"], "eodhd")
    if cached is not None:
        return cached

    if CACHE_ONLY:
        print(f"  [WARN] --cache-only set but no cache for {t['display']} "
              f"| {t['label']} — returning empty")
        return []

    return fetch_eodhd(t["display"], t["symbol"], window)


# ---------------------------------------------------------------------------
# Per-ticker pipeline: fetch → C4 score → C5 filter → C5 score
# ---------------------------------------------------------------------------

def run_ticker(t: dict) -> dict:
    """
    Full pipeline for one ticker. Returns result dict with all metrics.
    """
    display = t["display"]
    window  = {"label": t["label"], "start": t["start"], "end": t["end"]}

    print(f"\n  [{display}]  {t['name']}  |  {t['label']}")

    # --- Fetch ---
    articles = fetch_for_ticker(t)
    n_raw    = len(articles)

    if n_raw == 0:
        print(f"    NO ARTICLES — skipping scoring")
        return {
            "display":       display,
            "label":         t["label"],
            "returns":       t["returns"],
            "n_raw":         0,
            "n_filtered":    0,
            "retention_pct": None,
            "mean_sim":      None,
            "c4_score":      None,
            "c5_score":      None,
            "gpt_score":     t["gpt_score"],
            "n_articles_gpt": t["n_articles_gpt"],
            "fallback_used": False,
        }

    # --- Config 4: unfiltered FinBERT ---
    c4_scores  = _score_articles(articles)
    c4_mean    = sum(c4_scores) / len(c4_scores)
    print(f"    C4 (unfiltered) : articles={n_raw:>4}  score={c4_mean:+.4f}")

    # --- Config 5: cosine filter then FinBERT ---
    # Temporarily inject this ticker's reference query into the module's dict
    import method5_finbert_filtered as m5
    original_queries = dict(m5.REFERENCE_QUERIES)
    m5.REFERENCE_QUERIES[display] = REFERENCE_QUERIES[display]

    filtered, sims = filter_articles(display, articles, threshold=config.COSINE_THRESHOLD)
    n_filtered     = len(filtered)
    fallback_used  = False

    if n_filtered == 0:
        print(f"    cosine filter removed all articles — using unfiltered fallback")
        filtered      = articles
        sims          = [0.0] * n_raw
        n_filtered    = n_raw
        fallback_used = True

    c5_scores   = _score_articles(filtered)
    c5_mean     = sum(c5_scores) / len(c5_scores)
    mean_sim    = sum(sims) / len(sims) if sims else 0.0
    retention   = (n_filtered / n_raw * 100) if n_raw > 0 else 0.0

    # Restore original reference queries
    m5.REFERENCE_QUERIES.clear()
    m5.REFERENCE_QUERIES.update(original_queries)

    fb_marker = " [FALLBACK]" if fallback_used else ""
    print(
        f"    C5 (filtered)   : kept={n_filtered:>4}/{n_raw}  "
        f"({retention:.0f}%)  mean_sim={mean_sim:.3f}  "
        f"score={c5_mean:+.4f}{fb_marker}"
    )
    print(f"    GPT 5.1 score   : {t['gpt_score']:+.4f}")

    return {
        "display":        display,
        "label":          t["label"],
        "returns":        t["returns"],
        "n_raw":          n_raw,
        "n_filtered":     n_filtered,
        "retention_pct":  round(retention, 1),
        "mean_sim":       round(mean_sim, 4),
        "c4_score":       round(c4_mean, 4),
        "c5_score":       round(c5_mean, 4),
        "gpt_score":      t["gpt_score"],
        "n_articles_gpt": t["n_articles_gpt"],
        "fallback_used":  fallback_used,
    }


# ---------------------------------------------------------------------------
# Sentiment label helper
# ---------------------------------------------------------------------------

def _label(score) -> str:
    if score is None:
        return "N/A"
    if score >= 0.35:
        return "Bullish"
    if score >= 0.10:
        return "Somewhat Bullish"
    if score > -0.10:
        return "Neutral"
    if score > -0.35:
        return "Somewhat Bearish"
    return "Bearish"


def _direction_match(c5, gpt) -> str:
    """Do C5 and GPT 5.1 agree on direction (sign)?"""
    if c5 is None or gpt is None:
        return "N/A"
    same_sign = (c5 >= 0) == (gpt >= 0)
    return "✓ Agree" if same_sign else "✗ Differ"


# ---------------------------------------------------------------------------
# Print results tables
# ---------------------------------------------------------------------------

def print_results(results: list[dict]):
    TW  = 6    # ticker col
    DW  = 20   # date window col
    RW  = 9    # returns col
    NW  = 7    # n_articles col
    SW  = 12   # score col
    LW  = 18   # label col
    MW  = 10   # match col

    sep_main = (
        "+" + "-"*TW + "+" + "-"*DW + "+" + "-"*RW +
        "+" + "-"*NW + "+" + "-"*SW + "+" + "-"*SW +
        "+" + "-"*MW + "+"
    )

    # ---- Table 1: Core scores (matches slide deck columns) ----
    print(f"\n{'='*90}")
    print("  EODHD + Pre-filtering + FinBERT  vs  EYQ GPT 5.1")
    print("  (Headline column for slide deck: 'EODHD + pre-filtering + FinBERT' = C5 score)")
    print(f"{'='*90}")
    print(sep_main)
    print(
        "|" + "Ticker".center(TW) +
        "|" + "Date Window".center(DW) +
        "|" + "Returns".center(RW) +
        "|" + "N (EODHD)".center(NW) +
        "|" + "C5 FinBERT".center(SW) +
        "|" + "GPT 5.1".center(SW) +
        "|" + "Direction".center(MW) + "|"
    )
    print(sep_main)

    for r in results:
        c5_str  = f"{r['c5_score']:+.4f}" if r["c5_score"] is not None else "N/A"
        if r.get("fallback_used"):
            c5_str += "*"
        gpt_str = f"{r['gpt_score']:+.4f}"
        match   = _direction_match(r["c5_score"], r["gpt_score"])

        print(
            "|" + r["display"].center(TW) +
            "|" + r["label"].center(DW) +
            "|" + r["returns"].center(RW) +
            "|" + str(r["n_raw"]).center(NW) +
            "|" + c5_str.center(SW) +
            "|" + gpt_str.center(SW) +
            "|" + match.center(MW) + "|"
        )

    print(sep_main)
    print("  * fallback: cosine filter removed all articles; unfiltered used.")

    # ---- Table 2: Sentiment labels ----
    sep_lbl = (
        "+" + "-"*TW + "+" + "-"*DW +
        "+" + "-"*LW + "+" + "-"*LW + "+"
    )
    print(f"\n  Sentiment Labels")
    print(sep_lbl)
    print(
        "|" + "Ticker".center(TW) +
        "|" + "Date Window".center(DW) +
        "|" + "C5 FinBERT".center(LW) +
        "|" + "GPT 5.1".center(LW) + "|"
    )
    print(sep_lbl)
    for r in results:
        print(
            "|" + r["display"].center(TW) +
            "|" + r["label"].center(DW) +
            "|" + _label(r["c5_score"]).center(LW) +
            "|" + _label(r["gpt_score"]).center(LW) + "|"
        )
    print(sep_lbl)

    # ---- Table 3: Article coverage detail ----
    sep_cov = (
        "+" + "-"*TW + "+" + "-"*NW + "+" + "-"*NW +
        "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*12 + "+"
    )
    print(f"\n  Article Coverage & Filter Stats")
    print(sep_cov)
    print(
        "|" + "Ticker".center(TW) +
        "|" + "Raw".center(NW) +
        "|" + "Kept".center(NW) +
        "|" + "Retention".center(11) +
        "|" + "Mean Sim".center(11) +
        "|" + "C4 (unfilt)".center(12) + "|"
    )
    print(sep_cov)
    for r in results:
        ret_str  = f"{r['retention_pct']:.0f}%" if r["retention_pct"] is not None else "—"
        sim_str  = f"{r['mean_sim']:.3f}"        if r["mean_sim"]      is not None else "—"
        c4_str   = f"{r['c4_score']:+.4f}"       if r["c4_score"]      is not None else "N/A"
        print(
            "|" + r["display"].center(TW) +
            "|" + str(r["n_raw"]).center(NW) +
            "|" + str(r["n_filtered"]).center(NW) +
            "|" + ret_str.center(11) +
            "|" + sim_str.center(11) +
            "|" + c4_str.center(12) + "|"
        )
    print(sep_cov)

    # ---- Summary note for slide deck ----
    print(f"\n{'='*90}")
    print("  Summary for slide deck")
    print(f"{'='*90}")
    agree  = sum(1 for r in results
                 if r["c5_score"] is not None and
                 _direction_match(r["c5_score"], r["gpt_score"]) == "✓ Agree")
    total  = sum(1 for r in results if r["c5_score"] is not None)
    print(f"  Direction agreement (FinBERT C5 vs GPT 5.1): {agree}/{total} tickers")
    print()
    for r in results:
        if r["c5_score"] is None:
            continue
        delta = r["c5_score"] - r["gpt_score"]
        print(
            f"  {r['display']:<6} C5={r['c5_score']:+.4f}  "
            f"GPT={r['gpt_score']:+.4f}  "
            f"delta={delta:+.4f}  "
            f"({_direction_match(r['c5_score'], r['gpt_score'])})"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*90)
    print("  FinSent — LLM vs FinBERT Comparison Run")
    print("  EODHD + pre-filtering + FinBERT  (C5)  vs  EYQ GPT 5.1")
    print("="*90)

    # Warm up sentence-transformer before the loop so load time is not charged
    # to the first ticker
    print("\n[Setup] Pre-loading models …")
    get_st_model()

    results = []
    t_total = time.time()

    for t in TICKERS:
        t0 = time.time()
        result = run_ticker(t)
        results.append(result)
        print(f"    done in {time.time()-t0:.1f}s")

    print(f"\n[Total runtime] {time.time()-t_total:.1f}s")

    print_results(results)


if __name__ == "__main__":
    main()