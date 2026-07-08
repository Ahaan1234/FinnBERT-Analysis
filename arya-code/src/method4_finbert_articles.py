"""
method4_finbert_articles.py
===========================
Config 4 — News + FinBERT (no pre-filtering).

Pipeline:
  1. Fetch all articles for each ticker via the region-routed fetcher.
  2. Concatenate title + content for each article.
  3. Run every article through FinBERT (batched).
  4. Return the simple mean of per-article scores.
     (No relevance weighting — that is Config 5's job.)

Score = mean(P(positive) - P(negative)) across all articles, ∈ [-1, +1]
If zero articles are returned, score is reported as None (NaN sentinel).

Known limitation (documented):
  Articles are not filtered for ticker relevance before scoring.
  A single dominant article (e.g. an earnings report) can distort the mean.
  This is the controlled 'unfiltered' baseline against which Config 5 is compared.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda OpenMP fix

import config
from fetch_articles import fetch_articles
from method3_finbert_no_data import get_finbert, _scores_to_scalar


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def _score_articles(articles: list[dict]) -> list[float]:
    """
    Run FinBERT over a list of article dicts.
    Each article is scored on title + ' ' + content (truncated to max_length tokens).
    Returns a list of float scores parallel to `articles`.
    """
    clf = get_finbert()

    # Build text inputs — title + content, stripped to a safe char limit
    char_limit = config.FINBERT_MAX_LENGTH * 4   # rough chars per token ≈ 4
    texts = []
    for art in articles:
        combined = f"{art['title']}. {art['content']}"
        texts.append(combined[:char_limit])

    scores = []
    # Process in batches to avoid OOM on large article sets
    bs = config.FINBERT_BATCH_SIZE
    for i in range(0, len(texts), bs):
        batch   = texts[i : i + bs]
        results = clf(batch, truncation=True, max_length=config.FINBERT_MAX_LENGTH)
        for result in results:
            # pipeline top_k=None → list of lists; single item → list of dicts
            if isinstance(result, list) and isinstance(result[0], dict):
                scores.append(_scores_to_scalar(result))
            elif isinstance(result, list) and isinstance(result[0], list):
                scores.append(_scores_to_scalar(result[0]))
            else:
                scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Config 4 runner
# ---------------------------------------------------------------------------

def run_config4(window: dict) -> dict[str, float | None]:
    """
    Fetch articles for each ticker and return mean FinBERT score.

    Parameters
    ----------
    window : dict with keys 'label', 'start', 'end'

    Returns
    -------
    dict  {ticker_display: score | None}
          None means no articles were available for that ticker/window.
    """
    print(f"\n[Config 4] Window: {window['label']}")
    results = {}

    for ticker_display in config.TICKER_ORDER:
        articles = fetch_articles(ticker_display, window)

        if not articles:
            print(f"  {ticker_display:<12} NO ARTICLES — score=None")
            results[ticker_display] = None
            continue

        per_article_scores = _score_articles(articles)

        if not per_article_scores:
            results[ticker_display] = None
            continue

        mean_score = sum(per_article_scores) / len(per_article_scores)
        results[ticker_display] = round(mean_score, 4)

        print(
            f"  {ticker_display:<12} "
            f"articles={len(articles):>3}  "
            f"mean_score={mean_score:+.4f}  "
            f"[min={min(per_article_scores):+.4f}, max={max(per_article_scores):+.4f}]"
        )

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for window in config.DATE_WINDOWS:
        scores = run_config4(window)
        print(f"\nConfig 4 results — {window['label']}:")
        for t, s in scores.items():
            val = f"{s:+.4f}" if s is not None else "N/A"
            print(f"  {t:<12} {val}")