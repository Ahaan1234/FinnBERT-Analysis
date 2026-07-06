"""
method3_finbert_no_data.py
==========================
Config 3 — FinBERT Only (baseline, no article data).

Feeds FinBERT only the company name + ticker as a synthetic sentence.
This is the zero-information baseline: any score here reflects only
the semantic prior baked into FinBERT's training vocabulary.

Expected output: near-zero scores for most tickers.
That is correct and expected — documented as a known baseline behaviour.

Returns
-------
dict  {ticker_display: float}   sentiment score in [-1, +1]
      positive = bullish, negative = bearish
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/Anaconda OpenMP fix

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import config


# ---------------------------------------------------------------------------
# Load FinBERT once (module-level singleton)
# ---------------------------------------------------------------------------

def _load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL)
    device    = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,                  # return all three class scores
        truncation=True,
        max_length=config.FINBERT_MAX_LENGTH,
    )
    return clf


_finbert = None

def get_finbert():
    global _finbert
    if _finbert is None:
        print("[FinBERT] Loading model …")
        _finbert = _load_finbert()
        print("[FinBERT] Model ready.")
    return _finbert


# ---------------------------------------------------------------------------
# Score a single text → float in [-1, +1]
# ---------------------------------------------------------------------------

LABEL_SIGN = {"positive": +1.0, "negative": -1.0, "neutral": 0.0}

def _scores_to_scalar(label_score_list: list[dict]) -> float:
    """
    Convert FinBERT's list of {label, score} dicts to a single scalar.
    scalar = P(positive) - P(negative)   ∈ [-1, +1]
    """
    prob = {d["label"].lower(): d["score"] for d in label_score_list}
    return prob.get("positive", 0.0) - prob.get("negative", 0.0)


def score_text(text: str) -> float:
    clf    = get_finbert()
    result = clf(text[:config.FINBERT_MAX_LENGTH * 4])  # rough char limit
    # pipeline with top_k=None returns [[{label, score}, …]]
    if isinstance(result[0], list):
        return _scores_to_scalar(result[0])
    return _scores_to_scalar(result)


# ---------------------------------------------------------------------------
# Config 3 runner
# ---------------------------------------------------------------------------

# Human-readable company names for the synthetic prompt
COMPANY_NAMES = {
    "JPM":      "JPMorgan Chase (JPM)",
    "AVGO":     "Broadcom (AVGO)",
    "HDFCBANK": "HDFC Bank (HDFCBANK)",
    "Toyota":   "Toyota Motor (7203.T)",
    "Sony":     "Sony Group (6758.T)",
}

def run_config3(window: dict) -> dict[str, float]:
    """
    For each ticker feed a minimal synthetic sentence to FinBERT.
    No news articles used.

    Parameters
    ----------
    window : dict with keys 'label', 'start', 'end'

    Returns
    -------
    dict  {ticker_display: score}
    """
    print(f"\n[Config 3] Window: {window['label']}")
    results = {}

    for ticker_display in config.TICKER_ORDER:
        company = COMPANY_NAMES.get(ticker_display, ticker_display)
        # Minimal prompt — just the name. No date or market context.
        prompt = f"Financial sentiment for {company} stock."
        score  = score_text(prompt)
        results[ticker_display] = round(score, 4)
        print(f"  {ticker_display:<12} score={score:+.4f}")

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for window in config.DATE_WINDOWS:
        scores = run_config3(window)
        print(f"\nConfig 3 results — {window['label']}:")
        for t, s in scores.items():
            print(f"  {t:<12} {s:+.4f}")