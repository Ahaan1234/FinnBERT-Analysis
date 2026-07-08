"""
cache_utils.py
==============
Thin read/write wrapper around a local JSON cache.

Cache file per (ticker, window_label, source):
    cache/<ticker>__<window_label_slug>__<source>.json

On cache hit  → returns parsed list of article dicts immediately.
On cache miss → caller fetches from API, then calls save_cache().

This prevents re-burning free-tier quota on repeated benchmark runs.
"""

import json
import os
import re

from config import CACHE_DIR


def _slug(text: str) -> str:
    """Convert an arbitrary string to a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)


def _cache_path(ticker: str, window_label: str, source: str) -> str:
    filename = f"{_slug(ticker)}__{_slug(window_label)}__{_slug(source)}.json"
    return os.path.join(CACHE_DIR, filename)


def load_cache(ticker: str, window_label: str, source: str):
    """
    Returns list[dict] if a valid cache file exists, else None.
    """
    path = _cache_path(ticker, window_label, source)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"  [cache HIT]  {ticker} | {window_label} | {source} "
                  f"({len(data)} articles)")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [cache WARN] corrupt cache for {ticker}/{window_label}/{source}: {exc}")
            return None
    return None


def save_cache(ticker: str, window_label: str, source: str, articles: list):
    """
    Persist articles (list of dicts) to a JSON file.
    Silent on failure — cache is best-effort.
    """
    path = _cache_path(ticker, window_label, source)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(articles, fh, ensure_ascii=False, indent=2)
        print(f"  [cache SAVE] {ticker} | {window_label} | {source} "
              f"({len(articles)} articles) → {os.path.basename(path)}")
    except OSError as exc:
        print(f"  [cache WARN] could not save cache: {exc}")


def clear_cache(ticker: str = None, window_label: str = None, source: str = None):
    """
    Utility: delete matching cache files.
    Pass None to wildcard-match that dimension.
    """
    deleted = 0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        parts = fname[:-5].split("__")   # strip .json, split on __
        if len(parts) != 3:
            continue
        t, w, s = parts
        match = (
            (ticker       is None or _slug(ticker)       == t) and
            (window_label is None or _slug(window_label) == w) and
            (source       is None or _slug(source)       == s)
        )
        if match:
            os.remove(os.path.join(CACHE_DIR, fname))
            deleted += 1
    print(f"[cache CLEAR] {deleted} file(s) removed.")