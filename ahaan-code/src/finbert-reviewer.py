# =============================================================================
# finbert_reviewer_v2.py
#
# Scores each news article for a ticker on two axes:
#   1. FinBERT sentiment  – bullish (+1) vs bearish (-1) signal
#   2. Relevance          – composite of five signals:
#        a. Symbol crowding   : how focused the article is on this ticker
#        b. Title mention     : whether the ticker/company appears in the title
#        c. Content density   : normalised mention-frequency in the body
#        d. Cosine similarity : semantic proximity to a rich financial query
#        e. Recency           : exponential decay from the most recent article
#
# Relevance-weighted sentiment is then computed across all articles.
# =============================================================================

import ast
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKER = "AAPL"
COMPANY_NAME = "Apple"
RECENCY_HALF_LIFE_DAYS = 7

# Aliases recognised as referring to the same company/ticker
TICKER_ALIASES: dict[str, set[str]] = {
    "AAPL": {"AAPL", "APC"},
}

# Weights must sum to 1.0
RELEVANCE_WEIGHTS: dict[str, float] = {
    "symbol_crowding":   0.25,
    "title_mention":     0.05,
    "content_density":   0.25,
    "cosine_similarity": 0.35,
    "recency":           0.10,
}

# Rich semantic query used to anchor cosine-similarity relevance
SEMANTIC_QUERY = (
    f"{COMPANY_NAME} ({TICKER}) stock market financial news earnings revenue "
    "guidance analyst sentiment products services AI China supply chain"
)

# FinBERT encodes up to 512 tokens; SentenceTransformer batch size for GPU/CPU
FINBERT_MAX_LENGTH   = 512
SBERT_BATCH_SIZE     = 64          # tune upward on a GPU with plenty of VRAM
SBERT_MAX_CHARACTERS = 10_000      # truncate article text before encoding

# I/O paths
NEWS_DATA_PATH = "ahaan-code/results/AAPL_US_data.csv"
OUTPUT_PATH    = "ahaan-code/results/AAPL_US_data_finbert.csv"

# Model identifiers
FINBERT_MODEL_ID = "ProsusAI/finbert"
SBERT_MODEL_ID   = "all-MiniLM-L6-v2"

# =============================================================================
# DEVICE SELECTION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {DEVICE}")

# =============================================================================
# MODEL LOADING
# =============================================================================

print("[models] Loading FinBERT …")
sentiment_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
sentiment_model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)
sentiment_model.to(DEVICE).eval()
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

print("[models] Loading SentenceTransformer …")
sbert_model = SentenceTransformer(SBERT_MODEL_ID, device=str(DEVICE))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_finbert(text: str) -> dict[str, float] | None:
    """
    Run FinBERT on a single article text.

    Returns a dict with keys 'positive', 'negative', 'neutral', and
    'sentiment_score' (= positive – negative), or None when text is empty.
    """
    if not (isinstance(text, str) and text.strip()):
        return None

    tokens = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=FINBERT_MAX_LENGTH,
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        logits = sentiment_model(**tokens).logits

    probs = F.softmax(logits, dim=-1)[0]
    scores = {label: probs[i].item() for i, label in enumerate(SENTIMENT_LABELS)}
    scores["sentiment_score"] = scores["positive"] - scores["negative"]
    return scores


def embed_texts_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings in batches using SentenceTransformer.

    Returns an (N, D) float32 array of L2-normalised embeddings.
    """
    return sbert_model.encode(
        texts,
        batch_size=SBERT_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # gives unit vectors → dot product = cosine sim
        convert_to_numpy=True,
    )


def cosine_similarity_to_query(article_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each article embedding and the query.

    Because both sets of embeddings are L2-normalised, this is just a dot
    product, avoiding any redundant division.

    Returns a 1-D array of shape (N,).
    """
    # query_embedding shape: (D,) → broadcast across rows of article_embeddings
    return article_embeddings @ query_embedding          # (N,)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """
    Min-max scale an array to [0, 1].

    If all values are identical the array is returned as all-zeros (avoid NaN).
    """
    lo, hi = values.min(), values.max()
    spread = hi - lo
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / spread


def compute_symbol_crowding(symbols_text: str, aliases: set[str]) -> float:
    """
    Return 1 / (1 + number_of_other_companies) based on the symbols column.

    A score of 1.0 means the article is tagged to this ticker only.
    """
    try:
        symbols: list[str] = ast.literal_eval(symbols_text)
    except (ValueError, SyntaxError):
        symbols = []

    other_roots: set[str] = set()
    for sym in symbols:
        root = sym.split(".")[0].upper()
        root = re.sub(r"\d+", "", root) or root
        if root not in aliases:
            other_roots.add(root)

    return 1.0 / (1.0 + len(other_roots))


def compute_recency(article_date: pd.Timestamp, reference_date: pd.Timestamp, half_life_days: float) -> float:
    """
    Exponential decay score: 1.0 for the most recent article, decaying
    toward 0 for older ones.
    """
    if pd.isna(article_date):
        return 0.0
    age_days = (reference_date - article_date).total_seconds() / 86_400.0
    return 0.5 ** (age_days / half_life_days)


def compute_relevance(
    symbol_crowding:  float,
    title_mention:    float,
    content_density:  float,
    cosine_sim:       float,
    recency:          float,
    weights:          dict[str, float],
) -> float:
    """
    Weighted sum of all five relevance signals.
    """
    return (
        weights["symbol_crowding"]   * symbol_crowding
        + weights["title_mention"]   * title_mention
        + weights["content_density"] * content_density
        + weights["cosine_similarity"] * cosine_sim
        + weights["recency"]         * recency
    )

# =============================================================================
# DATA LOADING
# =============================================================================

print(f"[data] Reading {NEWS_DATA_PATH} …")
articles_df   = pd.read_csv(NEWS_DATA_PATH)
article_dates = pd.to_datetime(articles_df["date"], utc=True, errors="coerce")
most_recent   = article_dates.max()

# Pre-compile mention regex for this ticker + company name
mention_regex = re.compile(
    rf"\b({re.escape(TICKER)}|{re.escape(COMPANY_NAME)})\b",
    re.IGNORECASE,
)
aliases_for_ticker = TICKER_ALIASES.get(TICKER, {TICKER})

# =============================================================================
# PASS 1 – PER-ARTICLE SIGNALS THAT DO NOT REQUIRE NORMALISATION
# =============================================================================

print("[pass 1] Computing per-article signals …")

finbert_labels:        list[str | None]   = []
pos_scores:            list[float | None] = []
neg_scores:            list[float | None] = []
neu_scores:            list[float | None] = []
sentiment_scores:      list[float | None] = []
symbol_crowding_scores: list[float]       = []
title_mention_scores:  list[float]        = []
content_density_raw:   list[float]        = []
recency_scores_list:   list[float]        = []
combined_texts:        list[str]          = []   # for SBERT encoding

for idx, row in articles_df.iterrows():
    text    = row["content"]
    title   = row["title"]
    symbols = row["symbols"]
    date    = article_dates[idx]

    # ── FinBERT ──────────────────────────────────────────────────────────────
    fb = run_finbert(text)
    if fb is not None:
        finbert_labels.append(max(SENTIMENT_LABELS, key=lambda l: fb[l]))
        pos_scores.append(fb["positive"])
        neg_scores.append(fb["negative"])
        neu_scores.append(fb["neutral"])
        sentiment_scores.append(fb["sentiment_score"])
    else:
        finbert_labels.append(None)
        pos_scores.append(None)
        neg_scores.append(None)
        neu_scores.append(None)
        sentiment_scores.append(None)

    # ── Symbol crowding ───────────────────────────────────────────────────────
    symbol_crowding_scores.append(compute_symbol_crowding(symbols, aliases_for_ticker))

    # ── Title mention ─────────────────────────────────────────────────────────
    has_title_mention = bool(mention_regex.search(title)) if isinstance(title, str) else False
    title_mention_scores.append(1.0 if has_title_mention else 0.0)

    # ── Content density (raw, will be normalised below) ───────────────────────
    if isinstance(text, str) and text.strip():
        word_count   = max(len(text.split()), 1)
        mention_hits = len(mention_regex.findall(text))
        content_density_raw.append(mention_hits / word_count)
    else:
        content_density_raw.append(0.0)

    # ── Recency ───────────────────────────────────────────────────────────────
    recency_scores_list.append(compute_recency(date, most_recent, RECENCY_HALF_LIFE_DAYS))

    # ── Build combined text for SBERT (title + content, truncated) ───────────
    title_str   = title if isinstance(title, str) else ""
    content_str = text  if isinstance(text,  str) else ""
    combined    = (title_str + " " + content_str).strip()
    combined_texts.append(combined[:SBERT_MAX_CHARACTERS])

# =============================================================================
# NORMALISATION – Content Density
# =============================================================================

content_density_arr        = np.array(content_density_raw, dtype=np.float32)
normalized_content_density = minmax_normalize(content_density_arr)

# =============================================================================
# PASS 2 – SEMANTIC COSINE SIMILARITY (BATCHED)
# =============================================================================

print("[pass 2] Encoding articles with SentenceTransformer …")
article_embeddings = embed_texts_batch(combined_texts)          # (N, D)

print("[pass 2] Encoding query …")
query_embedding = sbert_model.encode(
    SEMANTIC_QUERY,
    normalize_embeddings=True,
    convert_to_numpy=True,
)                                                               # (D,)

raw_cosine_scores      = cosine_similarity_to_query(article_embeddings, query_embedding)
normalized_cosine_scores = minmax_normalize(raw_cosine_scores)

# =============================================================================
# PASS 3 – RELEVANCE SCORES & WEIGHTED SENTIMENT
# =============================================================================

print("[pass 3] Computing relevance and weighted sentiment …")

relevance_scores: list[float] = []
for i in range(len(articles_df)):
    r = compute_relevance(
        symbol_crowding=  symbol_crowding_scores[i],
        title_mention=    title_mention_scores[i],
        content_density=  float(normalized_content_density[i]),
        cosine_sim=       float(normalized_cosine_scores[i]),
        recency=          recency_scores_list[i],
        weights=          RELEVANCE_WEIGHTS,
    )
    relevance_scores.append(r)

# Relevance-weighted overall sentiment (same formula as original pipeline)
total_weighted_sentiment = 0.0
total_relevance_weight   = 0.0
for sent, rel in zip(sentiment_scores, relevance_scores):
    if sent is not None:
        total_weighted_sentiment += sent * rel
        total_relevance_weight   += rel

if total_relevance_weight > 0:
    rw_sentiment = total_weighted_sentiment / total_relevance_weight
else:
    rw_sentiment = 0.0

# =============================================================================
# OUTPUT – Attach all columns and save
# =============================================================================

articles_df["finbert_sentiment"]      = finbert_labels
articles_df["finbert_positive"]       = pos_scores
articles_df["finbert_negative"]       = neg_scores
articles_df["finbert_neutral"]        = neu_scores
articles_df["sentiment_score"]        = sentiment_scores
articles_df["symbol_crowding_score"]  = symbol_crowding_scores
articles_df["title_mention_score"]    = title_mention_scores
articles_df["content_density_score"]  = normalized_content_density
articles_df["cosine_similarity_score"]= raw_cosine_scores        # raw cosine ∈ [-1, 1]
articles_df["normalized_cosine_score"]= normalized_cosine_scores  # min-max ∈ [0, 1]
articles_df["recency_score"]          = recency_scores_list
articles_df["relevance_score"]        = relevance_scores

articles_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[output] Saved to {OUTPUT_PATH}")

# Summary
print(articles_df[[
    "date", "title",
    "finbert_sentiment", "sentiment_score",
    "normalized_cosine_score", "relevance_score",
]])
print(f"\nRelevance-weighted overall sentiment for {TICKER}: {rw_sentiment:.4f}")