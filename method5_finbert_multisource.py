"""
method5_finbert_multisource.py
-------------------------------
METHOD 5: FinBERT on multi-source news with cosine-similarity pre-filtering.

Pipeline:
  1. Collect articles from TWO sources per ticker/week:
       - Alpha Vantage NEWS_SENTIMENT  (US tickers: good; India/Japan: weak)
       - GDELT DOC 2.0 API            (global; free; noisy)

  2. Deduplicate by title similarity (simple Jaccard on word sets).

  3. Pre-filter using cosine similarity:
       - Embed each article (title + summary) with a lightweight
         sentence-transformers model (all-MiniLM-L6-v2).
       - Embed a query: "{company_name} stock market news {ticker}".
       - Keep top-N articles by cosine similarity score.
       - Also apply a hard keyword filter: drop articles with no
         mention of the company name or ticker in headline/summary.

  4. Run FinBERT on the filtered top-N articles.

  5. Aggregate: recency-weighted mean, same as Method 4.

  This is the approach Dilip approved: summaries are good to work with,
  and the pre-filtering step is the highest-ROI quality improvement.

Constants you can tune:
  TOP_N_ARTICLES   — how many articles to keep after cosine filter (default 8)
  MIN_COSINE_SIM   — drop anything below this similarity threshold (default 0.15)

Usage:
  python method5_finbert_multisource.py

Output:
  results/method5_finbert_multisource.csv
  results/method5_finbert_multisource.json
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import csv
import time
import warnings
import re
warnings.filterwarnings("ignore")

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

from config import (
    TICKERS, VOLATILE_WEEKS, FINBERT_MODEL,
    ALPHA_VANTAGE_API_KEY, OUTPUT_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tunable constants ──────────────────────────────────────────────────────────
TOP_N_ARTICLES = 8      # how many articles survive cosine filtering
MIN_COSINE_SIM = 0.15   # hard floor: below this we always drop

# ── Load models ────────────────────────────────────────────────────────────────
print(f"Loading FinBERT from '{FINBERT_MODEL}' …")
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
model.eval()
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

print("Loading sentence-transformer (all-MiniLM-L6-v2) …")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Models ready.\n")


# ── FinBERT scorer ─────────────────────────────────────────────────────────────
def finbert_score(text: str) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True,
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


# ── Source 1: Alpha Vantage ────────────────────────────────────────────────────
def fetch_av_articles(ticker: str, date_from: str, date_to: str) -> list[dict]:
    """Fetch from Alpha Vantage, return normalised article dicts."""
    av_ticker = ticker.split(".")[0]
    t_from = date_from.replace("-", "") + "T0000"
    t_to   = date_to.replace("-", "")   + "T2359"

    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={av_ticker}"
        f"&time_from={t_from}&time_to={t_to}"
        f"&limit=50&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        feed = data.get("feed", [])
    except Exception:
        feed = []

    normalised = []
    for art in feed:
        normalised.append({
            "source":    f"AlphaVantage:{art.get('source', '')}",
            "title":     art.get("title", ""),
            "summary":   art.get("summary", ""),
            "published": art.get("time_published", ""),
        })
    return normalised


# ── Source 2: GDELT DOC 2.0 ───────────────────────────────────────────────────
def fetch_gdelt_articles(company: str, date_from: str, date_to: str) -> list[dict]:
    """
    Query GDELT DOC 2.0 for articles mentioning the company name.
    GDELT is free, no key required, but is noisy — pre-filtering handles this.
    Returns normalised article dicts.
    """
    # GDELT date format: YYYYMMDDHHMMSS
    gd_from = date_from.replace("-", "") + "000000"
    gd_to   = date_to.replace("-", "")   + "235959"

    # URL-encode company name (spaces → %20)
    query = requests.utils.quote(f'"{company}" financial news stock')

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={query}"
        f"&mode=artlist"
        f"&maxrecords=25"
        f"&startdatetime={gd_from}"
        f"&enddatetime={gd_to}"
        f"&format=json"
    )
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
        articles = data.get("articles", [])
    except Exception:
        articles = []

    normalised = []
    for art in articles:
        normalised.append({
            "source":    f"GDELT:{art.get('domain', '')}",
            "title":     art.get("title", ""),
            "summary":   art.get("seendate", "") + " " + art.get("title", ""),
            "published": art.get("seendate", ""),
        })
    return normalised


# ── Deduplication ──────────────────────────────────────────────────────────────
def jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(re.findall(r'\w+', a.lower()))
    sb = set(re.findall(r'\w+', b.lower()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(articles: list[dict], threshold: float = 0.7) -> list[dict]:
    """Remove near-duplicate articles using title Jaccard similarity."""
    unique = []
    for art in articles:
        is_dup = False
        for u in unique:
            if jaccard(art["title"], u["title"]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(art)
    return unique


# ── Hard keyword filter ────────────────────────────────────────────────────────
def keyword_filter(articles: list[dict], ticker: str, company: str) -> list[dict]:
    """
    Drop articles that contain neither the company name nor the ticker
    in their title + summary. This removes completely irrelevant articles
    that happened to be in the GDELT result set.
    """
    ticker_root = ticker.split(".")[0].lower()
    company_words = set(company.lower().split())

    def relevant(art: dict) -> bool:
        text = (art["title"] + " " + art["summary"]).lower()
        if ticker_root in text:
            return True
        # Check if at least one distinctive company word appears
        for word in company_words:
            if len(word) > 4 and word in text:
                return True
        return False

    return [a for a in articles if relevant(a)]


# ── Cosine similarity filter (the main pre-filter) ────────────────────────────
def cosine_filter(
    articles: list[dict],
    ticker: str,
    company: str,
    top_n: int = TOP_N_ARTICLES,
    min_sim: float = MIN_COSINE_SIM,
) -> list[dict]:
    """
    Embed each article and a ticker-specific query, keep the top-N
    by cosine similarity. Drops anything below min_sim threshold.
    """
    if not articles:
        return []

    # Build a rich query that describes what we want
    query = (
        f"{company} ({ticker}) stock market financial news sentiment "
        f"earnings revenue guidance analyst price"
    )

    texts = [
        f"{a['title']}. {a['summary']}".strip()
        for a in articles
    ]

    # Encode query and all articles in one batch
    query_emb   = embedder.encode(query,  convert_to_tensor=True)
    article_embs = embedder.encode(texts, convert_to_tensor=True, batch_size=32)

    cosine_scores = util.cos_sim(query_emb, article_embs)[0]  # shape [n_articles]

    scored = sorted(
        zip(articles, cosine_scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    # Attach cosine score and filter
    filtered = []
    for art, sim in scored[:top_n]:
        if sim < min_sim:
            break
        art["cosine_sim"] = round(float(sim), 4)
        filtered.append(art)

    return filtered


# ── Aggregate FinBERT scores ───────────────────────────────────────────────────
def aggregate_finbert(articles: list[dict]) -> dict:
    """Run FinBERT on each article and return aggregated (recency-weighted) result."""
    if not articles:
        return {
            "num_articles": 0,
            "sentiment_score": None,
            "positive_prob": None,
            "negative_prob": None,
            "neutral_prob": None,
            "predicted_label": None,
            "article_details": [],
        }

    n = len(articles)
    total_w = w_pos = w_neg = w_neu = w_score = 0.0
    details = []

    for i, art in enumerate(articles):
        # Recency weight (assuming articles are still in original time order)
        weight = 0.5 + 0.5 * (i / max(n - 1, 1))

        text = f"{art['title']}. {art['summary']}".strip()
        if not text or text == ".":
            continue

        fb = finbert_score(text)

        w_pos   += weight * fb["positive_prob"]
        w_neg   += weight * fb["negative_prob"]
        w_neu   += weight * fb["neutral_prob"]
        w_score += weight * fb["sentiment_score"]
        total_w += weight

        details.append({
            "title":        art["title"],
            "source":       art["source"],
            "published":    art["published"],
            "cosine_sim":   art.get("cosine_sim"),
            "finbert_score": fb["sentiment_score"],
            "finbert_label": fb["predicted_label"],
            "finbert_pos":   fb["positive_prob"],
            "finbert_neg":   fb["negative_prob"],
            "finbert_neu":   fb["neutral_prob"],
            "weight":        round(weight, 3),
        })

    if total_w == 0:
        return {
            "num_articles": n,
            "sentiment_score": None,
            "positive_prob": None,
            "negative_prob": None,
            "neutral_prob": None,
            "predicted_label": None,
            "article_details": details,
        }

    agg_pos   = round(w_pos   / total_w, 4)
    agg_neg   = round(w_neg   / total_w, 4)
    agg_neu   = round(w_neu   / total_w, 4)
    agg_score = round(w_score / total_w, 4)
    probs_avg = [agg_pos, agg_neg, agg_neu]
    pred_label = LABEL_MAP[int(probs_avg.index(max(probs_avg)))]

    return {
        "num_articles":    len(details),
        "sentiment_score": agg_score,
        "positive_prob":   agg_pos,
        "negative_prob":   agg_neg,
        "neutral_prob":    agg_neu,
        "predicted_label": pred_label,
        "article_details": details,
    }


# ── Main analysis ──────────────────────────────────────────────────────────────
all_results = []

print("=== METHOD 5: FinBERT — Multi-source + cosine similarity pre-filtering ===\n")
print(f"Settings: TOP_N={TOP_N_ARTICLES}, MIN_COSINE_SIM={MIN_COSINE_SIM}\n")

for week_key, week in VOLATILE_WEEKS.items():
    print(f"\n── {week['label']} ({week['start']} → {week['end']}) ──")

    for ticker, company in TICKERS.items():
        print(f"\n  {ticker} ({company})")

        # Step 1: collect from both sources
        av_arts   = fetch_av_articles(ticker, week["start"], week["end"])
        gdelt_arts = fetch_gdelt_articles(company, week["start"], week["end"])
        raw_count = len(av_arts) + len(gdelt_arts)
        print(f"    Raw: {len(av_arts)} AV + {len(gdelt_arts)} GDELT = {raw_count}")

        # Step 2: combine + dedup
        combined = deduplicate(av_arts + gdelt_arts, threshold=0.7)
        print(f"    After dedup: {len(combined)}")

        # Step 3a: hard keyword filter
        kw_filtered = keyword_filter(combined, ticker, company)
        print(f"    After keyword filter: {len(kw_filtered)}")

        # Step 3b: cosine similarity filter
        cos_filtered = cosine_filter(kw_filtered, ticker, company)
        print(f"    After cosine filter (top-{TOP_N_ARTICLES}): {len(cos_filtered)}")

        # Step 4: FinBERT
        result = aggregate_finbert(cos_filtered)

        score  = result["sentiment_score"]
        label  = result["predicted_label"]
        n_arts = result["num_articles"]
        print(f"    FinBERT → {n_arts} articles | score={score!r:>8} | label={label}")

        row = {
            "method":             "5_finbert_multisource",
            "week_key":           week_key,
            "week_label":         week["label"],
            "week_start":         week["start"],
            "week_end":           week["end"],
            "ticker":             ticker,
            "company":            company,
            "raw_article_count":  raw_count,
            "after_dedup":        len(combined),
            "after_kw_filter":    len(kw_filtered),
            "after_cosine_filter": len(cos_filtered),
            "num_articles_scored": n_arts,
            "sentiment_score":    score,
            "positive_prob":      result["positive_prob"],
            "negative_prob":      result["negative_prob"],
            "neutral_prob":       result["neutral_prob"],
            "predicted_label":    label,
            "top_n_setting":      TOP_N_ARTICLES,
            "min_cosine_sim":     MIN_COSINE_SIM,
            "article_details":    result["article_details"],
        }
        all_results.append(row)

        time.sleep(0.3)   # polite pacing between API calls

# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "method5_finbert_multisource.csv")
csv_fields = [
    "method", "week_key", "week_label", "week_start", "week_end",
    "ticker", "company",
    "raw_article_count", "after_dedup", "after_kw_filter",
    "after_cosine_filter", "num_articles_scored",
    "sentiment_score", "positive_prob", "negative_prob", "neutral_prob",
    "predicted_label", "top_n_setting", "min_cosine_sim",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for row in all_results:
        writer.writerow({k: row[k] for k in csv_fields})

# ── Save JSON ──────────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "method5_finbert_multisource.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
print("\nDone — Method 5 complete.")