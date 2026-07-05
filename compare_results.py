"""
compare_results.py
------------------
Load results from Methods 3, 4, and 5, merge them into a single
comparison table, and print a structured summary.

Also produces:
  results/comparison_table.csv   — wide format: one row per ticker×week
  results/comparison_summary.txt — human-readable summary

Run this AFTER all three method scripts have completed:
  python method3_finbert_no_data.py
  python method4_finbert_alphavantage.py
  python method5_finbert_multisource.py
  python compare_results.py

Usage:
  python compare_results.py
"""

import os
import json
import csv

from config import TICKERS, VOLATILE_WEEKS, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────────
def load_json(filename: str) -> list[dict]:
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — run the corresponding method script first.")
        return []
    with open(path) as f:
        return json.load(f)


m3 = load_json("method3_finbert_no_data.json")
m4 = load_json("method4_finbert_alphavantage.json")
m5 = load_json("method5_finbert_multisource.json")


# ── Build a lookup: (ticker, week_key) -> score ────────────────────────────────
# Method 3 has no week dependency — same score for both weeks.
m3_lookup: dict = {}
for row in m3:
    ticker = row["ticker"]
    m3_lookup[ticker] = {
        "score": row["sentiment_score"],
        "label": row["predicted_label"],
    }

m4_lookup: dict = {}
for row in m4:
    key = (row["ticker"], row["week_key"])
    m4_lookup[key] = {
        "score":       row["sentiment_score"],
        "label":       row["predicted_label"],
        "n_articles":  row["num_articles"],
        "av_score":    row["av_mean_sentiment"],
    }

m5_lookup: dict = {}
for row in m5:
    key = (row["ticker"], row["week_key"])
    m5_lookup[key] = {
        "score":      row["sentiment_score"],
        "label":      row["predicted_label"],
        "n_articles": row["num_articles_scored"],
        "raw_count":  row["raw_article_count"],
        "after_cosine": row["after_cosine_filter"],
    }


# ── Build comparison table ─────────────────────────────────────────────────────
comparison_rows = []

for week_key, week in VOLATILE_WEEKS.items():
    for ticker, company in TICKERS.items():

        m3_data = m3_lookup.get(ticker, {})
        m4_data = m4_lookup.get((ticker, week_key), {})
        m5_data = m5_lookup.get((ticker, week_key), {})

        row = {
            "week":          week_key,
            "week_label":    week["label"],
            "ticker":        ticker,
            "company":       company,
            # Method 3 — no-data baseline
            "m3_score":      m3_data.get("score"),
            "m3_label":      m3_data.get("label"),
            # Method 4 — AV + FinBERT
            "m4_n_articles": m4_data.get("n_articles"),
            "m4_score":      m4_data.get("score"),
            "m4_label":      m4_data.get("label"),
            "m4_av_score":   m4_data.get("av_score"),   # AV's own keyword score
            # Method 5 — multi-source + cosine + FinBERT
            "m5_raw_count":  m5_data.get("raw_count"),
            "m5_after_cos":  m5_data.get("after_cosine"),
            "m5_n_articles": m5_data.get("n_articles"),
            "m5_score":      m5_data.get("score"),
            "m5_label":      m5_data.get("label"),
            # Deltas vs baseline
            "m4_vs_m3":     (
                round(m4_data["score"] - m3_data["score"], 4)
                if m4_data.get("score") is not None and m3_data.get("score") is not None
                else None
            ),
            "m5_vs_m3":     (
                round(m5_data["score"] - m3_data["score"], 4)
                if m5_data.get("score") is not None and m3_data.get("score") is not None
                else None
            ),
            "m5_vs_m4":     (
                round(m5_data["score"] - m4_data["score"], 4)
                if m5_data.get("score") is not None and m4_data.get("score") is not None
                else None
            ),
        }
        comparison_rows.append(row)


# ── Save comparison CSV ────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "comparison_table.csv")
csv_fields = [
    "week", "week_label", "ticker", "company",
    "m3_score", "m3_label",
    "m4_n_articles", "m4_score", "m4_label", "m4_av_score",
    "m5_raw_count", "m5_after_cos", "m5_n_articles", "m5_score", "m5_label",
    "m4_vs_m3", "m5_vs_m3", "m5_vs_m4",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Saved: {csv_path}")


# ── Print human-readable summary ───────────────────────────────────────────────
def fmt_score(v) -> str:
    if v is None:
        return "  n/a  "
    return f"{v:+.4f}"

def fmt_label(v) -> str:
    if v is None:
        return "  ---  "
    return f"{v:>8}"

summary_lines = []
divider = "=" * 110

for week_key, week in VOLATILE_WEEKS.items():
    summary_lines.append(divider)
    summary_lines.append(f"  WEEK: {week['label']}")
    summary_lines.append(divider)
    header = (
        f"  {'Ticker':<18} {'Company':<30} "
        f"{'M3 (no data)':>14} "
        f"{'M4 (AV+FB)':>12} {'AV_kw':>8} "
        f"{'M5 (multi+cos)':>15} "
        f"{'Δ M4-M3':>9} {'Δ M5-M3':>9} {'Δ M5-M4':>9}"
    )
    summary_lines.append(header)
    summary_lines.append("-" * 110)

    for row in comparison_rows:
        if row["week"] != week_key:
            continue
        line = (
            f"  {row['ticker']:<18} {row['company']:<30} "
            f"{fmt_score(row['m3_score']):>14} "
            f"{fmt_score(row['m4_score']):>12} {fmt_score(row['m4_av_score']):>8} "
            f"{fmt_score(row['m5_score']):>15} "
            f"{fmt_score(row['m4_vs_m3']):>9} {fmt_score(row['m5_vs_m3']):>9} "
            f"{fmt_score(row['m5_vs_m4']):>9}"
        )
        summary_lines.append(line)
    summary_lines.append("")

summary_lines.append(divider)
summary_lines.append("  LEGEND")
summary_lines.append("  M3 (no data)   : FinBERT primed with ticker/company name only. No news. Null-hypothesis baseline.")
summary_lines.append("  M4 (AV+FB)     : FinBERT on Alpha Vantage news summaries (AV_kw = AV's own keyword score).")
summary_lines.append("  M5 (multi+cos) : FinBERT on Alpha Vantage + GDELT, after cosine-similarity pre-filtering.")
summary_lines.append("  Score range [-1, +1].  Δ = (method score) - (reference score).")
summary_lines.append("  Interpretation:")
summary_lines.append("    Δ M4-M3 > 0: news pulled sentiment more positive than no-data prior.")
summary_lines.append("    Δ M5-M4 > 0: multi-source + filtering moved score vs single-source.")
summary_lines.append("    During negative week, scores should be < 0; during positive week, > 0.")
summary_lines.append(divider)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

# Save to file
txt_path = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
with open(txt_path, "w") as f:
    f.write(summary_text)
print(f"\nSaved: {txt_path}")
print("\nDone — comparison complete.")