# Pulls news articles for a ticker + date window and saves them to CSV so
# downstream scripts (finbert_reviewer.py, run.py) can score them.
#
# EODHD doesn't cover the Tokyo Stock Exchange, so any ".T"-suffixed ticker
# goes through yfinance instead - it only returns a title + short summary,
# not a full article body, but that's still enough for FinBERT. yfinance's
# news feed also has no date-range filter and caps out around 200 articles
# total, so it always just pulls the most recent ones it'll give us.

import os
import time
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
EODHD_API_KEY = os.environ["EODHD_API_KEY"]


class ContentRetriever:
    """Fetches and saves news articles for one ticker over one date window."""

    YFINANCE_ARTICLE_COUNT = 200  # yfinance caps out around here regardless of what you ask for

    def __init__(self, ticker, company, date_from, date_to, label, run_eodhd=True, output_path=None):
        self.ticker = ticker
        self.company = company
        self.date_from = date_from
        self.date_to = date_to
        self.label = label
        self.run_eodhd = run_eodhd
        self.output_path = output_path or f"ahaan-code/results/saved_news/{ticker}_US_data_{label}.csv"

    def uses_yfinance(self):
        return self.ticker.endswith(".T")

    def already_retrieved(self):
        return Path(self.output_path).exists()

    def fetch_eodhd_articles(self):
        # EODHD's `from`/`to` params silently return 0 results for some
        # tickers (e.g. SAP.XETRA) even though articles exist in that
        # window - fetching unfiltered and trimming client-side avoids
        # the bug entirely and matches what a manual unfiltered URL returns.
        params = {
            "s": self.ticker,
            "limit": 1000,
            "api_token": EODHD_API_KEY,
            "fmt": "json",
        }
        r = requests.get("https://eodhd.com/api/news", params=params, timeout=30)
        r.raise_for_status()
        df = pd.json_normalize(r.json())
        if df.empty:
            return df

        dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
        window_start = pd.Timestamp(self.date_from, tz="UTC")
        window_end = pd.Timestamp(self.date_to, tz="UTC") + pd.Timedelta(days=1)
        return df[(dates >= window_start) & (dates < window_end)].reset_index(drop=True)

    def fetch_yfinance_articles(self):
        news_items = yf.Ticker(self.ticker).get_news(count=self.YFINANCE_ARTICLE_COUNT, tab="all")
        rows = []
        for item in news_items:
            content = item.get("content", {})
            rows.append({
                "date": content.get("pubDate", ""),
                "title": content.get("title", ""),
                "content": content.get("summary", ""),
                "link": content.get("canonicalUrl", {}).get("url", ""),
                "symbols": [self.ticker],
            })
        return pd.DataFrame(rows)

    def retrieve(self):
        """Fetches articles and saves them to self.output_path. Returns the DataFrame."""
        print(f"\n=== {self.ticker} ({self.company}), {self.date_from} to {self.date_to} ===")

        if self.uses_yfinance():
            df = self.fetch_yfinance_articles()
        else:
            if not self.run_eodhd:
                print("  skipping - run_eodhd is False, set it to True to spend an API call")
                return None
            df = self.fetch_eodhd_articles()
            time.sleep(0.5)  # polite pacing between eodhd calls

        df.to_csv(self.output_path, index=False)
        print(f"  saved {len(df)} articles to {self.output_path}")
        return df


# Undated legacy pulls - kept around from before the pipeline moved to
# per-week windows. Not used by run.py; run this file directly to refresh them.
LEGACY_TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "7203.T": "Toyota",
    "6758.T": "Sony",
}
LEGACY_EODHD_FROM = "2024-08-01"
LEGACY_EODHD_TO = "2025-07-28"

if __name__ == "__main__":
    for ticker, company in LEGACY_TICKERS.items():
        retriever = ContentRetriever(
            ticker=ticker,
            company=company,
            date_from=LEGACY_EODHD_FROM,
            date_to=LEGACY_EODHD_TO,
            label="legacy",
            # legacy files live directly under results/, not results/saved_news/
            output_path=f'ahaan-code/results/{ticker.replace(".","_")}_data.csv',
        )
        retriever.retrieve()
