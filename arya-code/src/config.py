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
# Maps display name → (ticker_symbol, region, data_source)
TICKERS = {
    "SETBY":     ("SETBY",       "US",      "eodhd"),
    "Alibaba":   ("9988.HK",     "HongKong","eodhd"),
    "MU":        ("MU",          "US",      "eodhd"),
    "TM":        ("TM",          "US",      "eodhd"),
    "HMC":       ("HMC",         "US",      "eodhd"),
    "NOVO-B":    ("NOVO-B.CO",   "Denmark", "eodhd"),
    "SAP":       ("SAP.XETRA",   "Germany", "eodhd"),
    "FDX":       ("FDX",         "US",      "eodhd"),
    "Baidu":     ("9888.HK",     "HongKong","eodhd"),
    "SONY":      ("SONY",        "US",      "eodhd"),
}

# Convenience list ordered for display tables
TICKER_ORDER = [
    "SETBY",
    "Alibaba",
    "MU",
    "TM",
    "HMC",
    "NOVO-B",
    "SAP",
    "FDX",
    "Baidu",
    "SONY",
]

# ---------------------------------------------------------------------------
# DATE WINDOWS  — inclusive of both endpoints (00:00 → 23:59 UTC)
# ---------------------------------------------------------------------------
DATE_WINDOWS = [
    {
        "label": "17–26 Jun 2026",
        "start": "2026-06-17",
        "end":   "2026-06-26",
    },
    {
        "label": "6–15 May 2026",
        "start": "2026-05-06",
        "end":   "2026-05-15",
    },
    {
        "label": "17–26 Jun 2026",
        "start": "2026-06-17",
        "end":   "2026-06-26",
    },
    {
        "label": "1–8 May 2026",
        "start": "2026-05-01",
        "end":   "2026-05-08",
    },
    {
        "label": "6–13 May 2026",
        "start": "2026-05-06",
        "end":   "2026-05-13",
    },
    {
        "label": "15–22 Jun 2026",
        "start": "2026-06-15",
        "end":   "2026-06-22",
    },
    {
        "label": "17–24 Apr 2026",
        "start": "2026-04-17",
        "end":   "2026-04-24",
    },
    {
        "label": "16–23 Jun 2026",
        "start": "2026-06-16",
        "end":   "2026-06-23",
    },
    {
        "label": "1–8 Jul 2026",
        "start": "2026-07-01",
        "end":   "2026-07-08",
    },
    {
        "label": "6–13 May 2026",
        "start": "2026-05-06",
        "end":   "2026-05-13",
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