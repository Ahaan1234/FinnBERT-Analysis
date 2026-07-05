"""
config.py
---------
Shared configuration for all sentiment analysis scripts.

Volatile weeks (from basket_weekly_returns.csv analysis):
  - Max POSITIVE:  week ending 2024-08-16  (avg basket return: +6.45%)
  - Max NEGATIVE:  week ending 2025-04-04  (avg basket return: -9.72%)

Stocks: 3 US, 4 India, 3 Japan — exactly matching Atharva's selection.
"""

# ── Volatile week windows ──────────────────────────────────────────────────────
# We use the calendar week that *contains* the Friday close date.
VOLATILE_WEEKS = {
    "positive": {
        "label": "Max-Positive (week ending 2024-08-16)",
        "start": "2024-08-12",   # Monday
        "end":   "2024-08-16",   # Friday
    },
    "negative": {
        "label": "Max-Negative (week ending 2025-04-04)",
        "start": "2025-03-31",   # Monday
        "end":   "2025-04-04",   # Friday
    },
}

# ── Tickers ────────────────────────────────────────────────────────────────────
TICKERS = {
    # US
    "JPM":          "JPMorgan Chase",
    "TSLA":         "Tesla",
    "AVGO":         "Broadcom",
    # India
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services",
    "HDFCBANK.NS":  "HDFC Bank",
    "INFY.NS":      "Infosys",
    # Japan
    "7203.T":       "Toyota Motor",
    "9984.T":       "SoftBank Group",
    "6758.T":       "Sony Group",
}

# ── Alpha Vantage ──────────────────────────────────────────────────────────────
# Replace with your own key from alphavantage.co (free, 25 req/day)
ALPHA_VANTAGE_API_KEY = "8FC34M8EB64WHXL5"   # <-- REPLACE THIS

# ── FinBERT model ──────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = "results"