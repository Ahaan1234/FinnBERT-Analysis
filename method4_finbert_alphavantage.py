"""
method4_finbert_alphavantage.py
--------------------------------
METHOD 4: FinBERT sentiment on news articles fetched from Alpha Vantage.

Pipeline:
  1. For each ticker × volatile week, pull news from Alpha Vantage
     NEWS_SENTIMENT endpoint (articles published within the week window).
  2. Run every article summary through FinBERT.
  3. Aggregate per-ticker: weighted mean sentiment score (recency-weighted).
  4. Compare FinBERT scores with Alpha Vantage's own keyword-based scores —
     this shows whether running a real model matters vs their heuristic.

Alpha Vantage free tier: 25 req/day.
  - 10 tickers × 2 weeks = 20 requests.  Fits within free tier.
  - The API returns up to 50 articles per call in the date range.

Usage:
  python method4_finbert_alphavantage.py

  Set ALPHA_VANTAGE_API_KEY in config.py before running.

Output:
  results/method4_finbert_alphavantage.csv
  results/method4_finbert_alphavantage.json
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import csv
import time
import warnings
warnings.filterwarnings("ignore")

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    TICKERS, VOLATILE_WEEKS, FINBERT_MODEL,
    ALPHA_VANTAGE_API_KEY, OUTPUT_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load FinBERT ───────────────────────────────────────────────────────────────
print(f"Loading FinBERT from '{FINBERT_MODEL}' …")
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
model.eval()
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


def finbert_score(text: str) -> dict:
    """Score a single text with FinBERT. Returns probs + composite score."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    return {
        "positive_prob":   round(probs[0], 4),
        "negative_prob":   round(probs[1], 4),
        "neutral_prob":    round(probs[2], 4),
        "predicted_label": LABEL_MAP[pred_idx],
        "sentiment_score": round(probs[0] - probs[1], 4),
    }


def fetch_av_news(ticker: str, date_from: str, date_to: str) -> list[dict]:
    """
    Fetch news from Alpha Vantage NEWS_SENTIMENT endpoint.

    Alpha Vantage ticker format: US tickers pass as-is (JPM, TSLA, AVGO).
    Indian/Japanese tickers (RELIANCE.NS, 7203.T) are NOT supported by AV
    — we handle this gracefully and return an empty list with a note.

    date_from / date_to: format 'YYYY-MM-DD'
    Returns list of article dicts.
    """
    # Alpha Vantage only reliably covers US tickers in its news endpoint.
    # Indian (.NS) and Japanese (.T) tickers will return empty feeds.
    av_ticker = ticker.split(".")[0]   # strip exchange suffix

    # Convert date format to AV's expected YYYYMMDDTHHMM
    t_from = date_from.replace("-", "") + "T0000"
    t_to   = date_to.replace("-", "")   + "T2359"

    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={av_ticker}"
        f"&time_from={t_from}"
        f"&time_to={t_to}"
        f"&limit=50"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [AV fetch error for {ticker}]: {e}")
        return []

    if "feed" not in data:
        # Rate-limited or no results
        note = data.get("Information", data.get("Note", "No feed returned"))
        print(f"    [AV] {ticker}: {note[:80]}")
        return []

    return data["feed"]


def articles_to_finbert(articles: list[dict], ticker: str) -> dict:
    """
    Run FinBERT on each article's summary, then aggregate.

    We use recency weighting: articles published closer to the end of the
    week get slightly higher weight (linear decay, oldest = 0.5, newest = 1.0).
    Also records AV's own keyword-based relevance_score for comparison.
    """
    if not articles:
        return {
            "num_articles": 0,
            "sentiment_score": None,
            "positive_prob": None,
            "negative_prob": None,
            "neutral_prob": None,
            "predicted_label": None,
            "av_mean_sentiment": None,
            "article_details": [],
        }

    n = len(articles)
    article_details = []

    total_weight    = 0.0
    w_pos = w_neg = w_neu = w_score = 0.0
    av_scores = []

    for i, art in enumerate(articles):
        # Linear recency weight: oldest article gets 0.5, newest gets 1.0
        weight = 0.5 + 0.5 * (i / max(n - 1, 1))

        # Text to score: title + summary (summary is more informative for FinBERT)
        text = f"{art.get('title', '')}. {art.get('summary', '')}".strip()
        if not text or text == ".":
            continue

        fb = finbert_score(text)

        # Alpha Vantage's own sentiment for this ticker (if present)
        av_ticker_score = None
        av_relevance    = None
        for ts in art.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.split(".")[0].upper():
                try:
                    av_ticker_score = float(ts.get("ticker_sentiment_score", 0))
                    av_relevance    = float(ts.get("relevance_score", 0))
                except (ValueError, TypeError):
                    pass
                break

        # Only include article if AV relevance >= 0.3 (basic relevance gate)
        # For tickers without AV relevance data, include everything.
        effective_relevance = av_relevance if av_relevance is not None else 1.0
        if effective_relevance < 0.3:
            continue

        w_pos   += weight * fb["positive_prob"]
        w_neg   += weight * fb["negative_prob"]
        w_neu   += weight * fb["neutral_prob"]
        w_score += weight * fb["sentiment_score"]
        total_weight += weight

        if av_ticker_score is not None:
            av_scores.append(av_ticker_score)

        article_details.append({
            "title":              art.get("title", ""),
            "published":          art.get("time_published", ""),
            "source":             art.get("source", ""),
            "av_relevance":       av_relevance,
            "av_ticker_score":    av_ticker_score,
            "finbert_score":      fb["sentiment_score"],
            "finbert_label":      fb["predicted_label"],
            "finbert_pos":        fb["positive_prob"],
            "finbert_neg":        fb["negative_prob"],
            "finbert_neu":        fb["neutral_prob"],
            "weight":             round(weight, 3),
        })

    if total_weight == 0:
        return {
            "num_articles": n,
            "sentiment_score": None,
            "positive_prob": None,
            "negative_prob": None,
            "neutral_prob": None,
            "predicted_label": None,
            "av_mean_sentiment": None,
            "article_details": article_details,
        }

    agg_pos   = round(w_pos   / total_weight, 4)
    agg_neg   = round(w_neg   / total_weight, 4)
    agg_neu   = round(w_neu   / total_weight, 4)
    agg_score = round(w_score / total_weight, 4)
    probs_avg = [agg_pos, agg_neg, agg_neu]
    pred_label = LABEL_MAP[int(probs_avg.index(max(probs_avg)))]

    return {
        "num_articles":      len(article_details),
        "sentiment_score":   agg_score,
        "positive_prob":     agg_pos,
        "negative_prob":     agg_neg,
        "neutral_prob":      agg_neu,
        "predicted_label":   pred_label,
        "av_mean_sentiment": round(sum(av_scores) / len(av_scores), 4) if av_scores else None,
        "article_details":   article_details,
    }


# ── Main analysis ──────────────────────────────────────────────────────────────
all_results = []

print("\n=== METHOD 4: FinBERT on Alpha Vantage news feed ===\n")

for week_key, week in VOLATILE_WEEKS.items():
    print(f"\n── {week['label']} ({week['start']} → {week['end']}) ──")

    for ticker, company in TICKERS.items():
        print(f"  {ticker:18s} ({company}) … ", end="", flush=True)

        # Fetch news from Alpha Vantage
        articles = fetch_av_news(ticker, week["start"], week["end"])

        # Score with FinBERT
        result = articles_to_finbert(articles, ticker)

        n_arts   = result["num_articles"]
        score    = result["sentiment_score"]
        label    = result["predicted_label"]
        av_score = result["av_mean_sentiment"]

        print(f"{n_arts} articles  |  "
              f"FinBERT={score!r:>8} ({label})  |  AV={av_score!r}")

        row = {
            "method":            "4_finbert_alphavantage",
            "week_key":          week_key,
            "week_label":        week["label"],
            "week_start":        week["start"],
            "week_end":          week["end"],
            "ticker":            ticker,
            "company":           company,
            "num_articles":      n_arts,
            "sentiment_score":   score,
            "positive_prob":     result["positive_prob"],
            "negative_prob":     result["negative_prob"],
            "neutral_prob":      result["neutral_prob"],
            "predicted_label":   label,
            "av_mean_sentiment": av_score,
            "article_details":   result["article_details"],
        }
        all_results.append(row)

        # Be gentle with the free API: 0.5s between calls
        time.sleep(0.5)

# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "method4_finbert_alphavantage.csv")
csv_fields = [
    "method", "week_key", "week_label", "week_start", "week_end",
    "ticker", "company",
    "num_articles", "sentiment_score",
    "positive_prob", "negative_prob", "neutral_prob",
    "predicted_label", "av_mean_sentiment",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for row in all_results:
        writer.writerow({k: row[k] for k in csv_fields})

# ── Save JSON ──────────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "method4_finbert_alphavantage.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
print("\nDone — Method 4 complete.")