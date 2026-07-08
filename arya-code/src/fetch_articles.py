"""
fetch_articles.py
=================
Three fetchers — one per region — each returning a normalised list of dicts:

    {
        "title":     str,
        "content":   str,          # body text if available, else title
        "published": str,          # ISO date string "YYYY-MM-DD"
        "source":    str,          # provider name
        "url":       str,
    }

Source routing (from config.py):
    EODHD     → JPM, AVGO
    Marketaux → HDFCBANK.NS
    GDELT     → 7203.T (Toyota), 6758.T (Sony)

All fetchers:
  1. Check local JSON cache first (no API call if hit).
  2. On miss, fetch from API and save to cache before returning.
  3. Filter to the inclusive date window [start, end].
"""

import time
from datetime import datetime, timedelta, timezone

import requests

import config
from cache_utils import load_cache, save_cache


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _in_window(date_str: str, start: str, end: str) -> bool:
    """
    True if date_str (YYYY-MM-DD or ISO8601) falls within [start, end] inclusive.
    Tolerant of timestamps with time components.
    """
    try:
        # Normalise to date only
        date_only = date_str[:10]
        return start <= date_only <= end
    except (TypeError, ValueError):
        return False


def _norm(title: str, content: str, published: str, source: str, url: str) -> dict:
    title   = (title   or "").strip()
    content = (content or title).strip()   # fall back to title if no body
    return {
        "title":     title,
        "content":   content if content else title,
        "published": published[:10] if published else "",
        "source":    source,
        "url":       url or "",
    }


# ---------------------------------------------------------------------------
# EODHD fetcher  (US: JPM, AVGO)
# ---------------------------------------------------------------------------

def fetch_eodhd(ticker_display: str, ticker_symbol: str,
                window: dict) -> list[dict]:
    """
    Calls EODHD /api/news for ticker_symbol within window.
    Paginates by moving 'offset' forward until we exhaust articles or leave window.
    """
    cached = load_cache(ticker_display, window["label"], "eodhd")
    if cached is not None:
        return cached

    articles = []
    offset   = 0
    limit    = int(config.EODHD_NEWS_PARAMS.get("limit", 100))
    start    = window["start"]
    end      = window["end"]

    # EODHD accepts from/to as YYYY-MM-DD
    params = {
        **config.EODHD_NEWS_PARAMS,
        "s":    ticker_symbol,
        "from": start,
        "to":   end,
    }

    while True:
        params["offset"] = offset
        try:
            resp = requests.get(config.EODHD_NEWS_URL, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  [EODHD ERROR] {ticker_display} offset={offset}: {exc}")
            break

        if not batch:
            break

        for item in batch:
            pub = (item.get("date") or item.get("published") or "")[:10]
            if not _in_window(pub, start, end):
                continue
            articles.append(_norm(
                title     = item.get("title", ""),
                content   = item.get("content", "") or item.get("summary", ""),
                published = pub,
                source    = "EODHD",
                url       = item.get("link", "") or item.get("url", ""),
            ))

        if len(batch) < limit:
            break                 # last page
        offset += limit
        time.sleep(0.5)          # be polite to free-tier rate limits

    print(f"  [EODHD]  {ticker_display} | {window['label']} → {len(articles)} articles")
    save_cache(ticker_display, window["label"], "eodhd", articles)
    return articles


# ---------------------------------------------------------------------------
# Marketaux fetcher  (India: HDFCBANK.NS)
# ---------------------------------------------------------------------------

def fetch_marketaux(ticker_display: str, ticker_symbol: str,
                    window: dict) -> list[dict]:
    """
    Calls Marketaux /v1/news/all.
    Free tier: 100 req/day, max 10 articles per page — paginate carefully.
    Marketaux accepts 'symbols' (comma-separated) and 'published_after/before'.
    """
    cached = load_cache(ticker_display, window["label"], "marketaux")
    if cached is not None:
        return cached

    articles = []
    page     = 1
    start    = window["start"]
    end      = window["end"]

    # Marketaux wants ISO8601 timestamps
    published_after  = f"{start}T00:00:00"
    published_before = f"{end}T23:59:59"

    # Marketaux uses base ticker for NSE (no ".NS" suffix)
    symbol_clean = ticker_symbol.replace(".NS", "").replace(".BSE", "")

    params = {
        **config.MARKETAUX_NEWS_PARAMS,
        "symbols":          symbol_clean,
        "published_after":  published_after,
        "published_before": published_before,
        "page":             page,
    }

    while True:
        params["page"] = page
        try:
            resp = requests.get(config.MARKETAUX_NEWS_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [Marketaux ERROR] {ticker_display} page={page}: {exc}")
            break

        batch = data.get("data", [])
        if not batch:
            break

        for item in batch:
            pub = (item.get("published_at") or "")[:10]
            if not _in_window(pub, start, end):
                continue

            # Build content from description + full snippet if available
            body = item.get("description", "") or item.get("snippet", "")
            articles.append(_norm(
                title     = item.get("title", ""),
                content   = body,
                published = pub,
                source    = "Marketaux",
                url       = item.get("url", ""),
            ))

        meta       = data.get("meta", {})
        total_found = meta.get("found", 0)
        returned   = meta.get("returned", 0)
        if len(articles) >= total_found or returned < int(params.get("limit", 10)):
            break

        page += 1
        time.sleep(1.0)   # free tier — generous sleep to avoid 429

    print(f"  [Marketaux] {ticker_display} | {window['label']} → {len(articles)} articles")
    save_cache(ticker_display, window["label"], "marketaux", articles)
    return articles


# ---------------------------------------------------------------------------
# GDELT fetcher  (Japan: Toyota 7203.T, Sony 6758.T)
# ---------------------------------------------------------------------------

def _gdelt_query_string(company_name: str, start: str, end: str) -> str:
    """
    Build a GDELT DOC 2.0 query.
    GDELT timespan works on last N minutes OR a startdatetime/enddatetime pair.
    Format: YYYYMMDDHHMMSS
    """
    start_dt = f"{start.replace('-','')}000000"
    end_dt   = f"{end.replace('-','')}235959"
    return (
        f'"{company_name}" sourcelang:english '
        f"startdatetime:{start_dt} enddatetime:{end_dt}"
    )


def fetch_gdelt(ticker_display: str, ticker_symbol: str,
                window: dict) -> list[dict]:
    """
    Calls GDELT DOC 2.0 artlist endpoint.
    No authentication required.
    Returns normalised article list.
    """
    cached = load_cache(ticker_display, window["label"], "gdelt")
    if cached is not None:
        return cached

    company_name = config.GDELT_COMPANY_NAMES.get(ticker_display, ticker_display)
    start        = window["start"]
    end          = window["end"]

    params = {
        **config.GDELT_DOC_PARAMS,
        "query": _gdelt_query_string(company_name, start, end),
    }

    articles = []
    try:
        resp = requests.get(config.GDELT_DOC_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [GDELT ERROR] {ticker_display} | {window['label']}: {exc}")
        save_cache(ticker_display, window["label"], "gdelt", articles)
        return articles

    for item in (data.get("articles") or []):
        pub = (item.get("seendate") or "")
        # GDELT seendate format: "20260403T120000Z" or "20260403120000"
        if len(pub) >= 8:
            pub_date = f"{pub[0:4]}-{pub[4:6]}-{pub[6:8]}"
        else:
            pub_date = ""

        if pub_date and not _in_window(pub_date, start, end):
            continue

        articles.append(_norm(
            title     = item.get("title", ""),
            content   = item.get("title", ""),   # GDELT artlist has no body
            published = pub_date,
            source    = item.get("domain", "GDELT"),
            url       = item.get("url", ""),
        ))

    print(f"  [GDELT]  {ticker_display} | {window['label']} → {len(articles)} articles")
    save_cache(ticker_display, window["label"], "gdelt", articles)
    return articles


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def fetch_articles(ticker_display: str, window: dict) -> list[dict]:
    """
    Route to the correct fetcher based on config.TICKERS source field.
    Returns list of normalised article dicts.
    """
    ticker_symbol, _region, source = config.TICKERS[ticker_display]

    if source == "eodhd":
        return fetch_eodhd(ticker_display, ticker_symbol, window)
    elif source == "marketaux":
        return fetch_marketaux(ticker_display, ticker_symbol, window)
    elif source == "gdelt":
        return fetch_gdelt(ticker_display, ticker_symbol, window)
    else:
        raise ValueError(f"Unknown source '{source}' for ticker '{ticker_display}'")