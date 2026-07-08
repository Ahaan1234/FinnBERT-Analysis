# Pull news articles per ticker and save each to its own CSV so downstream
# scripts (finbert-reviewer.py etc.) can score them.
#
# EODHD doesn't cover the Tokyo Stock Exchange, so Japanese tickers (Sony,
# Toyota) go through yfinance instead - it only returns a title + short
# summary, not a full article body, but that's still enough for FinBERT.
#
# yfinance's news feed has no date-range filter and caps out around 200
# articles total (tried GDELT for real date-range control, but it rate-limits
# hard and kept coming back empty/429) - so YFINANCE_ARTICLE_COUNT just pulls
# as many of the most recent articles as it'll give us.
#
# Set RUN_EODHD to False to skip all EODHD calls entirely (yfinance is free,
# so those tickers always run) - flip it to True only when you actually want
# to spend API credits.

import sys
import time
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EODHD_API_KEY

RUN_EODHD = True
YFINANCE_ARTICLE_COUNT = 200  # yfinance caps out around here regardless of what you ask for

EODHD_FROM = "2024-08-01"
EODHD_TO = "2025-07-28"

TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "7203.T": "Toyota",
    "6758.T": "Sony",
}

# EODHD-only tickers with their own date window (Toyota uses its NYSE ADR
# "TM" here since EODHD/Alpha Vantage don't cover the Tokyo Stock Exchange
# ticker directly)
DATED_EODHD_TICKERS = {
    "INTC": {"company": "Intel", "from": "2026-04-27", "to": "2026-05-01", "label": "01May"},
    "TM": {"company": "Toyota (NYSE ADR)", "from": "2026-03-02", "to": "2026-03-06", "label": "06Mar"},
}


def uses_yfinance(ticker):
    return ticker.endswith(".T")


def fetch_eodhd_articles(ticker, date_from=EODHD_FROM, date_to=EODHD_TO):
    params = {
        "s": ticker,
        "from": date_from,
        "to": date_to,
        "limit": 1000,
        "api_token": EODHD_API_KEY,
        "fmt": "json",
    }
    r = requests.get("https://eodhd.com/api/news", params=params, timeout=30)
    r.raise_for_status()
    return pd.json_normalize(r.json())


def fetch_yfinance_articles(ticker):
    news_items = yf.Ticker(ticker).get_news(count=YFINANCE_ARTICLE_COUNT, tab="all")
    rows = []
    for item in news_items:
        content = item.get("content", {})
        rows.append({
            "date": content.get("pubDate", ""),
            "title": content.get("title", ""),
            "content": content.get("summary", ""),
            "link": content.get("canonicalUrl", {}).get("url", ""),
            "symbols": [ticker],
        })
    return pd.DataFrame(rows)


for ticker, company in TICKERS.items():
    print(f"\n=== {ticker} ({company}) ===")

    if uses_yfinance(ticker):
        df = fetch_yfinance_articles(ticker)
    else:
        if not RUN_EODHD:
            print("  skipping - RUN_EODHD is False, set it to True to spend an API call")
            continue
        df = fetch_eodhd_articles(ticker)
        time.sleep(0.5)  # polite pacing between eodhd calls

    output_path = f'ahaan-code/results/{ticker.replace(".","_")}_data.csv'
    df.to_csv(output_path, index=False)
    print(f"  saved {len(df)} articles to {output_path}")

    for count, row in enumerate(df.itertuples(), start=1):
        excerpt = (row.content or "")[:30]
        print(f"  {count}. title: {row.title}\n     excerpt: {excerpt}")

for ticker, window in DATED_EODHD_TICKERS.items():
    print(f"\n=== {ticker} ({window['company']}), {window['from']} to {window['to']} ===")

    if not RUN_EODHD:
        print("  skipping - RUN_EODHD is False, set it to True to spend an API call")
        continue

    df = fetch_eodhd_articles(ticker, date_from=window["from"], date_to=window["to"])
    time.sleep(0.5)  # polite pacing between eodhd calls

    output_path = f'ahaan-code/results/saved_news/{ticker}_US_data_{window["label"]}.csv'
    df.to_csv(output_path, index=False)
    print(f"  saved {len(df)} articles to {output_path}")
