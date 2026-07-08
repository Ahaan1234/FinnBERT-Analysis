"""
compare_results.py
==================
Orchestrator and reporter for the FinSent Sentiment Benchmark.

Runs Configs 3, 4, 5 across both date windows for all 5 tickers.
Configs 1 & 2 (EYQ Incubator) are deprioritised — placeholders shown as '—'.

Output: two printed tables (one per window) matching Sentiment_Analysis_v1.docx,
        plus a side-by-side summary and a delta table (Config 5 − Config 4).

Score encoding: float in [-1, +1]
  +1.0 = maximally bullish    −1.0 = maximally bearish    0.0 = neutral

Usage
-----
    python compare_results.py

    # Skip API fetches and use only cached data (safe for re-runs):
    python compare_results.py --cache-only

    # Clear cache before run:
    python compare_results.py --clear-cache
"""

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda OpenMP fix

import config
from method3_finbert_no_data   import run_config3
from method4_finbert_articles  import run_config4
from method5_finbert_filtered  import run_config5


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

CACHE_ONLY  = "--cache-only"  in sys.argv
CLEAR_CACHE = "--clear-cache" in sys.argv

if CLEAR_CACHE:
    from cache_utils import clear_cache
    clear_cache()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

COL_WIDTH   = 22   # width of each config column
TICK_WIDTH  = 12   # width of ticker column
DEPR_MARK   = "—"  # placeholder for deprioritised configs

def _fmt(val) -> str:
    """Format a score value for table display."""
    if val is None:
        return "N/A".center(COL_WIDTH)
    if isinstance(val, dict):
        # Config 5 returns a rich dict; extract score
        val = val.get("score")
        if val is None:
            return "N/A".center(COL_WIDTH)
    return f"{val:+.4f}".center(COL_WIDTH)


def _sentiment_label(val) -> str:
    """Convert scalar to human-readable label."""
    if val is None:
        return ""
    if isinstance(val, dict):
        val = val.get("score")
    if val is None:
        return ""
    if val >= 0.35:
        return "Bullish"
    if val >= 0.10:
        return "Somewhat Bullish"
    if val > -0.10:
        return "Neutral"
    if val > -0.35:
        return "Somewhat Bearish"
    return "Bearish"


def _hline(col_count: int) -> str:
    return "+" + ("-" * TICK_WIDTH) + ("+" + ("-" * COL_WIDTH)) * col_count + "+"


def _header_row(labels: list[str]) -> str:
    row = "|" + "Ticker".center(TICK_WIDTH)
    for lbl in labels:
        row += "|" + lbl[:COL_WIDTH].center(COL_WIDTH)
    row += "|"
    return row


def print_window_table(
    window_label: str,
    c3: dict, c4: dict, c5: dict,
):
    """Print a table for one window with all 5 configs (1 & 2 as placeholders)."""

    COL_LABELS = [
        "C1 EYQ Only",
        "C2 EYQ+News",
        "C3 FinBERT",
        "C4 News+FINBERT",
        "C5 Filt+FinBERT",
    ]

    print(f"\n{'='*80}")
    print(f"  Window: {window_label}")
    print(f"{'='*80}")
    print(_hline(5))
    print(_header_row(COL_LABELS))
    print(_hline(5))

    for ticker in config.TICKER_ORDER:
        row = "|" + ticker.center(TICK_WIDTH)
        row += "|" + DEPR_MARK.center(COL_WIDTH)   # C1
        row += "|" + DEPR_MARK.center(COL_WIDTH)   # C2
        row += "|" + _fmt(c3.get(ticker))           # C3
        row += "|" + _fmt(c4.get(ticker))           # C4

        c5_val = c5.get(ticker, {})
        score  = c5_val.get("score") if isinstance(c5_val, dict) else c5_val
        fb     = c5_val.get("fallback_used", False) if isinstance(c5_val, dict) else False
        score_str = "N/A" if score is None else f"{score:+.4f}"
        if fb:
            score_str += "*"
        row += "|" + score_str.center(COL_WIDTH)    # C5
        row += "|"
        print(row)

    print(_hline(5))
    print(f"  * fallback: cosine filter removed all articles; unfiltered set used.")


def print_sentiment_labels(
    window_label: str,
    c3: dict, c4: dict, c5: dict,
):
    """Print a companion label table (Bearish / Neutral / Bullish) for readability."""

    COL_LABELS = ["C3 FinBERT", "C4 News+FinBERT", "C5 Filt+FinBERT"]
    LABEL_WIDTH = 20

    print(f"\n  Sentiment Labels — {window_label}")
    sep = "+" + ("-" * TICK_WIDTH) + ("+" + ("-" * LABEL_WIDTH)) * 3 + "+"
    print(sep)
    hdr = "|" + "Ticker".center(TICK_WIDTH)
    for lbl in COL_LABELS:
        hdr += "|" + lbl[:LABEL_WIDTH].center(LABEL_WIDTH)
    hdr += "|"
    print(hdr)
    print(sep)

    for ticker in config.TICKER_ORDER:
        c5_val = c5.get(ticker, {})
        c5_score = c5_val.get("score") if isinstance(c5_val, dict) else c5_val
        row = "|" + ticker.center(TICK_WIDTH)
        row += "|" + _sentiment_label(c3.get(ticker)).center(LABEL_WIDTH)
        row += "|" + _sentiment_label(c4.get(ticker)).center(LABEL_WIDTH)
        row += "|" + _sentiment_label(c5_score).center(LABEL_WIDTH)
        row += "|"
        print(row)

    print(sep)


def print_delta_table(window_label: str, c4: dict, c5: dict):
    """
    Print Config5 − Config4 delta.
    Positive delta = filtering made sentiment more positive (noise was negative).
    Negative delta = filtering made sentiment more negative (noise was positive).
    """
    print(f"\n  Config 5 − Config 4 delta (effect of cosine pre-filtering) — {window_label}")
    sep = "+" + ("-" * TICK_WIDTH) + "+" + ("-" * 14) + "+" + ("-" * 18) + "+"
    print(sep)
    print("|" + "Ticker".center(TICK_WIDTH) + "|" +
          "Delta".center(14) + "|" + "Interpretation".center(18) + "|")
    print(sep)

    for ticker in config.TICKER_ORDER:
        s4 = c4.get(ticker)
        c5_val = c5.get(ticker, {})
        s5 = c5_val.get("score") if isinstance(c5_val, dict) else c5_val

        if s4 is None or s5 is None:
            delta_str = "N/A"
            interp    = "—"
        else:
            delta     = s5 - s4
            delta_str = f"{delta:+.4f}"
            if abs(delta) < 0.01:
                interp = "No change"
            elif delta > 0:
                interp = "Filter ↑ bullish"
            else:
                interp = "Filter ↑ bearish"

        print("|" + ticker.center(TICK_WIDTH) + "|" +
              delta_str.center(14) + "|" + interp.center(18) + "|")

    print(sep)


def print_article_coverage(window_label: str, c5_results: dict):
    """Print article counts and filter retention rates from Config 5."""
    print(f"\n  Article coverage — {window_label}")
    sep = "+" + ("-" * TICK_WIDTH) + "+" + ("-"*8) + "+" + ("-"*8) + "+" + ("-"*12) + "+" + ("-"*12) + "+" + ("-"*10) + "+"
    print(sep)
    print("|" + "Ticker".center(TICK_WIDTH) +
          "|" + "Source".center(8) +
          "|" + "Raw".center(8) +
          "|" + "Kept".center(12) +
          "|" + "Retention".center(12) +
          "|" + "Mean Sim".center(10) + "|")
    print(sep)

    for ticker in config.TICKER_ORDER:
        _sym, _region, source = config.TICKERS[ticker]
        val        = c5_results.get(ticker, {})
        n_raw      = val.get("n_raw",      0)
        n_filtered = val.get("n_filtered", 0)
        mean_sim   = val.get("mean_sim",   0.0)
        retention  = f"{(n_filtered/n_raw*100):.0f}%" if n_raw > 0 else "—"

        print("|" + ticker.center(TICK_WIDTH) +
              "|" + source.upper().center(8) +
              "|" + str(n_raw).center(8) +
              "|" + str(n_filtered).center(12) +
              "|" + retention.center(12) +
              "|" + f"{mean_sim:.3f}".center(10) + "|")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*80)
    print("  FinSent Sentiment Benchmark — EY FSRM QTB")
    print("  Configs 3, 4, 5  |  5 tickers  |  2 windows")
    print("="*80)

    all_results = {}  # {window_label: {3: dict, 4: dict, 5: dict}}

    for window in config.DATE_WINDOWS:
        lbl = window["label"]
        print(f"\n{'─'*60}")
        print(f"  Running window: {lbl}")
        print(f"  [{window['start']}  →  {window['end']}]")
        print(f"{'─'*60}")

        t0 = time.time()

        c3 = run_config3(window)
        c4 = run_config4(window)
        c5 = run_config5(window)

        elapsed = time.time() - t0
        print(f"\n  [timing] {lbl} completed in {elapsed:.1f}s")

        all_results[lbl] = {3: c3, 4: c4, 5: c5}

    # -----------------------------------------------------------------------
    # Print tables for each window
    # -----------------------------------------------------------------------
    for window in config.DATE_WINDOWS:
        lbl = window["label"]
        r   = all_results[lbl]
        print_window_table(lbl, r[3], r[4], r[5])
        print_sentiment_labels(lbl, r[3], r[4], r[5])
        print_delta_table(lbl, r[4], r[5])
        print_article_coverage(lbl, r[5])

    # -----------------------------------------------------------------------
    # Side-by-side summary: C4 scores both windows
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  Side-by-side: Config 4 (News + FinBERT) scores across both windows")
    print(f"{'='*80}")
    w1, w2 = [w["label"] for w in config.DATE_WINDOWS]
    COL_W  = 20
    print("+" + "-"*TICK_WIDTH + "+" + "-"*COL_W + "+" + "-"*COL_W + "+")
    print("|" + "Ticker".center(TICK_WIDTH) +
          "|" + w1[:COL_W].center(COL_W) +
          "|" + w2[:COL_W].center(COL_W) + "|")
    print("+" + "-"*TICK_WIDTH + "+" + "-"*COL_W + "+" + "-"*COL_W + "+")
    for ticker in config.TICKER_ORDER:
        s1 = all_results[w1][4].get(ticker)
        s2 = all_results[w2][4].get(ticker)
        v1 = f"{s1:+.4f}" if s1 is not None else "N/A"
        v2 = f"{s2:+.4f}" if s2 is not None else "N/A"
        print("|" + ticker.center(TICK_WIDTH) + "|" + v1.center(COL_W) + "|" + v2.center(COL_W) + "|")
    print("+" + "-"*TICK_WIDTH + "+" + "-"*COL_W + "+" + "-"*COL_W + "+")

    print("\n[Done] Benchmark complete.\n")


if __name__ == "__main__":
    main()