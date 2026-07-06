"""
config.py
=========
Central configuration for the FinSent Sentiment Benchmark.

Covers:
  - Ticker universe and region routing
  - Date windows (inclusive both ends, UTC)
  - API keys (fill in before running)
  - Source assignment: EODHD → US, Marketaux → India, GDELT → Japan
  - Cache directory
  - FinBERT model choice
  - Cosine-similarity threshold for Config 5

Configs produced:
  3 — FinBERT, no article data   (baseline)
  4 — FinBERT + source articles  (raw)
  5 — FinBERT + source articles  (cosine pre-filtered)
"""

import os

# ---------------------------------------------------------------------------
# API KEYS  — set via environment variables or replace the empty strings
# ---------------------------------------------------------------------------
EODHD_API_KEY     = os.getenv("EODHD_API_KEY",    "6a4ba81bbf8452.26372332")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "4C7VKkspGngWXPTGHliB56C3EkhjUGAI1lsBuioo")
# GDELT is free / no key required

# ---------------------------------------------------------------------------
# TICKER UNIVERSE
# ---------------------------------------------------------------------------
# Maps display name → (ticker_symbol, region, data_source)
TICKERS = {
    "JPM":         ("JPM",          "US",    "eodhd"),
    "AVGO":        ("AVGO",         "US",    "eodhd"),
    "HDFCBANK":    ("HDFCBANK.NS",  "India", "marketaux"),
    "Toyota":      ("7203.T",       "Japan", "gdelt"),
    "Sony":        ("6758.T",       "Japan", "gdelt"),
}

# Convenience list ordered for display tables
TICKER_ORDER = ["JPM", "AVGO", "HDFCBANK", "Toyota", "Sony"]

# ---------------------------------------------------------------------------
# DATE WINDOWS  — inclusive of both endpoints (00:00 → 23:59 UTC)
# ---------------------------------------------------------------------------
DATE_WINDOWS = [
    {
        "label": "3–10 Apr 2026",
        "start": "2026-04-03",
        "end":   "2026-04-10",
    },
    {
        "label": "17–23 Jun 2026",
        "start": "2026-06-17",
        "end":   "2026-06-23",
    },
]

# ---------------------------------------------------------------------------
# SOURCE CONFIG
# ---------------------------------------------------------------------------

# EODHD news endpoint
EODHD_NEWS_URL = "https://eodhd.com/api/news"
EODHD_NEWS_PARAMS = {
    "api_token": EODHD_API_KEY,
    "limit":     100,           # max articles per request
    "fmt":       "json",
}

# Marketaux news endpoint
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"
MARKETAUX_NEWS_PARAMS = {
    "api_token": MARKETAUX_API_KEY,
    "language":  "en",
    "limit":     10,            # free tier: max 10 per page
}

# GDELT DOC 2.0 — query endpoint (no key required)
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DOC_PARAMS = {
    "mode":       "artlist",
    "maxrecords": 75,
    "format":     "json",
    "sort":       "DateDesc",
}

# Company name aliases used as GDELT search terms (ticker symbols don't work)
GDELT_COMPANY_NAMES = {
    "Toyota": "Toyota Motor",
    "Sony":   "Sony Group",
}

# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FINBERT
# ---------------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"   # HuggingFace model id
FINBERT_BATCH_SIZE = 16
FINBERT_MAX_LENGTH = 512             # tokens; headlines fit; truncate long bodies

# ---------------------------------------------------------------------------
# CONFIG 5 — cosine similarity pre-filter
# ---------------------------------------------------------------------------
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
COSINE_THRESHOLD = 0.30              # articles below this are dropped

# ---------------------------------------------------------------------------
# CONFIGS ACTIVE IN THIS RUN
# ---------------------------------------------------------------------------
# 1 & 2 require EYQ Incubator (EY internal LLM) — deprioritised
ACTIVE_CONFIGS = [3, 4, 5]
CONFIG_LABELS = {
    3: "Config 3 — FinBERT Only",
    4: "Config 4 — News + FinBERT",
    5: "Config 5 — Filtered News + FinBERT",
}