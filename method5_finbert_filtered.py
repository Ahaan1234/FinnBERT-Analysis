"""
method5_finbert_filtered.py
============================
Config 5 — Filtered News + FinBERT.

Pipeline:
  1. Fetch articles (same source routing as Config 4).
  2. Embed each article (title + content) and a reference query for the ticker
     using sentence-transformers (all-MiniLM-L6-v2).
  3. Compute cosine similarity between each article embedding and the ticker
     reference embedding.
  4. Drop articles below COSINE_THRESHOLD (default 0.30).
  5. Run surviving articles through FinBERT (batched).
  6. Return mean score across survivors.

Reference query per ticker is a short, semantically rich description of the
company's core business — more informative than the ticker symbol alone.

If filtering leaves zero articles, fall back to Config 4 behaviour (all articles)
and flag the result with a '*' in the compare table.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda OpenMP fix

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from fetch_articles import fetch_articles
from method3_finbert_no_data import get_finbert, _scores_to_scalar
from method4_finbert_articles import _score_articles


# ---------------------------------------------------------------------------
# Ticker reference queries  — used as the cosine anchor
# ---------------------------------------------------------------------------

REFERENCE_QUERIES = {
    "JPM":      "JPMorgan Chase banking financial services earnings revenue credit",
    "AVGO":     "Broadcom semiconductor chips networking AI infrastructure earnings",
    "HDFCBANK": "HDFC Bank India retail banking loans NPA earnings RBI",
    "Toyota":   "Toyota Motor automobile car sales production EV hybrid Japan",
    "Sony":     "Sony Group electronics gaming PlayStation music entertainment Japan",
}


# ---------------------------------------------------------------------------
# Load sentence-transformer once
# ---------------------------------------------------------------------------

_st_model = None

def get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        print(f"[SentenceTransformer] Loading {config.SENTENCE_TRANSFORMER_MODEL} …")
        _st_model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        print("[SentenceTransformer] Model ready.")
    return _st_model


# ---------------------------------------------------------------------------
# Cosine similarity (pure numpy — no sklearn dependency needed)
# ---------------------------------------------------------------------------

def _cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Filter articles by cosine similarity to ticker reference query
# ---------------------------------------------------------------------------

def filter_articles(
    ticker_display: str,
    articles: list[dict],
    threshold: float = config.COSINE_THRESHOLD,
) -> tuple[list[dict], list[float]]:
    """
    Returns (filtered_articles, similarity_scores) for articles that pass threshold.
    Both lists are parallel.
    """
    if not articles:
        return [], []

    st = get_st_model()
    ref_query = REFERENCE_QUERIES.get(ticker_display, ticker_display)

    # Build text corpus
    char_limit = 512
    texts = [
        f"{a['title']}. {a['content']}"[:char_limit]
        for a in articles
    ]

    # Embed reference query and all article texts
    ref_emb      = st.encode(ref_query,  convert_to_numpy=True, normalize_embeddings=True)
    article_embs = st.encode(texts,      convert_to_numpy=True, normalize_embeddings=True,
                             batch_size=32, show_progress_bar=False)

    kept_articles = []
    kept_sims     = []

    for art, emb in zip(articles, article_embs):
        sim = _cosine_sim(ref_emb, emb)
        if sim >= threshold:
            kept_articles.append(art)
            kept_sims.append(sim)

    return kept_articles, kept_sims


# ---------------------------------------------------------------------------
# Config 5 runner
# ---------------------------------------------------------------------------

def run_config5(window: dict) -> dict[str, dict]:
    """
    Fetch → cosine-filter → FinBERT → mean score.

    Returns
    -------
    dict  {
        ticker_display: {
            "score":         float | None,
            "n_raw":         int,     # articles before filter
            "n_filtered":    int,     # articles after filter
            "fallback_used": bool,    # True if filter left 0 articles
            "mean_sim":      float,   # mean cosine sim of survivors (or 0)
        }
    }
    """
    print(f"\n[Config 5] Window: {window['label']}")
    results = {}

    for ticker_display in config.TICKER_ORDER:
        articles = fetch_articles(ticker_display, window)
        n_raw    = len(articles)

        if n_raw == 0:
            print(f"  {ticker_display:<12} NO ARTICLES — score=None")
            results[ticker_display] = {
                "score": None, "n_raw": 0, "n_filtered": 0,
                "fallback_used": False, "mean_sim": 0.0,
            }
            continue

        filtered, sims = filter_articles(ticker_display, articles)
        n_filtered     = len(filtered)
        fallback_used  = False

        if n_filtered == 0:
            # Fall back to unfiltered — flag it
            print(f"  {ticker_display:<12} cosine filter removed all articles "
                  f"(threshold={config.COSINE_THRESHOLD}) — using unfiltered fallback")
            filtered      = articles
            sims          = [0.0] * len(articles)
            n_filtered    = len(filtered)
            fallback_used = True

        per_article_scores = _score_articles(filtered)
        mean_score = sum(per_article_scores) / len(per_article_scores)
        mean_sim   = sum(sims) / len(sims) if sims else 0.0

        marker = " [FALLBACK]" if fallback_used else ""
        print(
            f"  {ticker_display:<12} "
            f"raw={n_raw:>3}  kept={n_filtered:>3}  "
            f"mean_sim={mean_sim:.3f}  "
            f"score={mean_score:+.4f}{marker}"
        )

        results[ticker_display] = {
            "score":         round(mean_score, 4),
            "n_raw":         n_raw,
            "n_filtered":    n_filtered,
            "fallback_used": fallback_used,
            "mean_sim":      round(mean_sim, 4),
        }

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for window in config.DATE_WINDOWS:
        results = run_config5(window)
        print(f"\nConfig 5 results — {window['label']}:")
        for t, r in results.items():
            s = r["score"]
            val = f"{s:+.4f}" if s is not None else "N/A"
            fb  = " *fallback*" if r["fallback_used"] else ""
            print(
                f"  {t:<12} {val}  "
                f"(raw={r['n_raw']}, kept={r['n_filtered']}, "
                f"mean_sim={r['mean_sim']:.3f}){fb}"
            )