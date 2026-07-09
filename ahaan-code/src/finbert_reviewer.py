# =============================================================================
# finbert_reviewer.py
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


class FinBertReviewer:
    """Scores a CSV of articles for one ticker on sentiment + relevance."""

    FINBERT_MODEL_ID = "ProsusAI/finbert"
    SBERT_MODEL_ID = "all-MiniLM-L6-v2"
    SENTIMENT_LABELS = ["positive", "negative", "neutral"]

    FINBERT_MAX_LENGTH = 512     # tokens
    SBERT_BATCH_SIZE = 64        # tune upward on a GPU with plenty of VRAM
    SBERT_MAX_CHARACTERS = 10_000  # truncate article text before encoding
    RECENCY_HALF_LIFE_DAYS = 7

    # Weights must sum to 1.0
    RELEVANCE_WEIGHTS = {
        "symbol_crowding":   0.25,
        "title_mention":     0.05,
        "content_density":   0.35,
        "cosine_similarity": 0.25,
        "recency":           0.10,
    }

    # Loaded once and shared across every instance so scoring several tickers
    # in one process doesn't reload FinBERT/SBERT each time.
    _sentiment_tokenizer = None
    _sentiment_model = None
    _sbert_model = None

    def __init__(self, ticker, company_name, aliases):
        self.ticker = ticker
        self.company_name = company_name
        self.aliases = aliases
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mention_regex = re.compile(
            rf"\b({re.escape(ticker)}|{re.escape(company_name)})\b",
            re.IGNORECASE,
        )
        self.semantic_query = (
            f"{company_name} ({ticker}) stock market financial news earnings revenue "
            "guidance analyst sentiment products services AI China supply chain"
        )

        self._load_models()

    @classmethod
    def _load_models(cls):
        if cls._sentiment_tokenizer is None:
            print("[models] Loading FinBERT …")
            cls._sentiment_tokenizer = AutoTokenizer.from_pretrained(cls.FINBERT_MODEL_ID)
            cls._sentiment_model = AutoModelForSequenceClassification.from_pretrained(cls.FINBERT_MODEL_ID)
        if cls._sbert_model is None:
            print("[models] Loading SentenceTransformer …")
            cls._sbert_model = SentenceTransformer(cls.SBERT_MODEL_ID)

    # -------------------------------------------------------------------
    # Per-article signals
    # -------------------------------------------------------------------

    def run_finbert(self, text):
        """
        Run FinBERT on a single article text.

        Returns a dict with keys 'positive', 'negative', 'neutral', and
        'sentiment_score' (= positive – negative), or None when text is empty.
        """
        if not (isinstance(text, str) and text.strip()):
            return None

        tokens = self._sentiment_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.FINBERT_MAX_LENGTH,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = self._sentiment_model(**tokens).logits

        probs = F.softmax(logits, dim=-1)[0]
        scores = {label: probs[i].item() for i, label in enumerate(self.SENTIMENT_LABELS)}
        scores["sentiment_score"] = scores["positive"] - scores["negative"]
        return scores

    def embed_texts_batch(self, texts):
        """Encode a list of strings in batches. Returns an (N, D) L2-normalised array."""
        return self._sbert_model.encode(
            texts,
            batch_size=self.SBERT_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,   # gives unit vectors → dot product = cosine sim
            convert_to_numpy=True,
        )

    @staticmethod
    def cosine_similarity_to_query(article_embeddings, query_embedding):
        """Both inputs are L2-normalised, so this dot product is cosine similarity."""
        return article_embeddings @ query_embedding

    @staticmethod
    def minmax_normalize(values):
        """Min-max scale an array to [0, 1]. All-equal input returns all-zeros (avoid NaN)."""
        lo, hi = values.min(), values.max()
        spread = hi - lo
        if spread < 1e-9:
            return np.zeros_like(values)
        return (values - lo) / spread

    def compute_symbol_crowding(self, symbols_text):
        """1 / (1 + number_of_other_companies) based on the symbols column."""
        try:
            symbols = ast.literal_eval(symbols_text)
        except (ValueError, SyntaxError):
            symbols = []

        other_roots = set()
        for sym in symbols:
            root = sym.split(".")[0].upper()
            root = re.sub(r"\d+", "", root) or root
            if root not in self.aliases:
                other_roots.add(root)

        return 1.0 / (1.0 + len(other_roots))

    @staticmethod
    def compute_recency(article_date, reference_date, half_life_days):
        """Exponential decay: 1.0 for the most recent article, decaying toward 0 for older ones."""
        if pd.isna(article_date):
            return 0.0
        age_days = (reference_date - article_date).total_seconds() / 86_400.0
        return 0.5 ** (age_days / half_life_days)

    def compute_relevance(self, symbol_crowding, title_mention, content_density, cosine_sim, recency):
        """Weighted sum of all five relevance signals."""
        w = self.RELEVANCE_WEIGHTS
        return (
            w["symbol_crowding"]   * symbol_crowding
            + w["title_mention"]   * title_mention
            + w["content_density"] * content_density
            + w["cosine_similarity"] * cosine_sim
            + w["recency"]         * recency
        )

    # -------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------

    def score(self, data_path, output_path):
        """
        Runs FinBERT + relevance scoring on every article in data_path,
        saves the scored CSV to output_path, and returns the
        relevance-weighted overall sentiment score.
        """
        print(f"[data] Reading {data_path} …")
        articles_df = pd.read_csv(data_path)
        article_dates = pd.to_datetime(articles_df["date"], utc=True, errors="coerce")
        most_recent = article_dates.max()

        print("[pass 1] Computing per-article signals …")
        finbert_labels, pos_scores, neg_scores, neu_scores, sentiment_scores = [], [], [], [], []
        symbol_crowding_scores, title_mention_scores = [], []
        content_density_raw, recency_scores_list, combined_texts = [], [], []

        for idx, row in articles_df.iterrows():
            text = row["content"]
            title = row["title"]
            symbols = row["symbols"]
            date = article_dates[idx]

            fb = self.run_finbert(text)
            if fb is not None:
                finbert_labels.append(max(self.SENTIMENT_LABELS, key=lambda l: fb[l]))
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

            symbol_crowding_scores.append(self.compute_symbol_crowding(symbols))

            has_title_mention = bool(self.mention_regex.search(title)) if isinstance(title, str) else False
            title_mention_scores.append(1.0 if has_title_mention else 0.0)

            if isinstance(text, str) and text.strip():
                word_count = max(len(text.split()), 1)
                mention_hits = len(self.mention_regex.findall(text))
                content_density_raw.append(mention_hits / word_count)
            else:
                content_density_raw.append(0.0)

            recency_scores_list.append(self.compute_recency(date, most_recent, self.RECENCY_HALF_LIFE_DAYS))

            title_str = title if isinstance(title, str) else ""
            content_str = text if isinstance(text, str) else ""
            combined = (title_str + " " + content_str).strip()
            combined_texts.append(combined[:self.SBERT_MAX_CHARACTERS])

        content_density_arr = np.array(content_density_raw, dtype=np.float32)
        normalized_content_density = self.minmax_normalize(content_density_arr)

        print("[pass 2] Encoding articles with SentenceTransformer …")
        article_embeddings = self.embed_texts_batch(combined_texts)

        print("[pass 2] Encoding query …")
        query_embedding = self._sbert_model.encode(
            self.semantic_query, normalize_embeddings=True, convert_to_numpy=True,
        )

        raw_cosine_scores = self.cosine_similarity_to_query(article_embeddings, query_embedding)
        normalized_cosine_scores = self.minmax_normalize(raw_cosine_scores)

        print("[pass 3] Computing relevance and weighted sentiment …")
        relevance_scores = []
        for i in range(len(articles_df)):
            relevance_scores.append(self.compute_relevance(
                symbol_crowding=symbol_crowding_scores[i],
                title_mention=title_mention_scores[i],
                content_density=float(normalized_content_density[i]),
                cosine_sim=float(normalized_cosine_scores[i]),
                recency=recency_scores_list[i],
            ))

        total_weighted_sentiment = 0.0
        total_relevance_weight = 0.0
        for sent, rel in zip(sentiment_scores, relevance_scores):
            if sent is not None:
                total_weighted_sentiment += sent * rel
                total_relevance_weight += rel
        rw_sentiment = total_weighted_sentiment / total_relevance_weight if total_relevance_weight > 0 else 0.0

        articles_df["finbert_sentiment"] = finbert_labels
        articles_df["finbert_positive"] = pos_scores
        articles_df["finbert_negative"] = neg_scores
        articles_df["finbert_neutral"] = neu_scores
        articles_df["sentiment_score"] = sentiment_scores
        articles_df["symbol_crowding_score"] = symbol_crowding_scores
        articles_df["title_mention_score"] = title_mention_scores
        articles_df["content_density_score"] = normalized_content_density
        articles_df["cosine_similarity_score"] = raw_cosine_scores
        articles_df["normalized_cosine_score"] = normalized_cosine_scores
        articles_df["recency_score"] = recency_scores_list
        articles_df["relevance_score"] = relevance_scores

        articles_df.to_csv(output_path, index=False)
        print(f"\n[output] Saved to {output_path}")

        print(articles_df[[
            "date", "title", "finbert_sentiment", "sentiment_score",
            "normalized_cosine_score", "relevance_score",
        ]])

        return rw_sentiment


# Same standalone TICKER_CONFIGS + ACTIVE_TICKER/ACTIVE_WEEK selection pattern
# as before - run this file directly to score one ticker+week combo.
TICKER_CONFIGS = {
    "AVGO": {
        "company_name": "Broadcom",
        "aliases": {"AVGO"},
        "data_paths": {
            "10Apr": "ahaan-code/results/saved_news/AVGO_US_data_10Apr.csv",
            "23Jun": "ahaan-code/results/saved_news/AVGO_US_data_23Jun.csv",
        },
    },
    "JPM": {
        "company_name": "JPMorgan Chase",
        "aliases": {"JPM"},
        "data_paths": {
            "10Apr": "ahaan-code/results/saved_news/JPM_US_data_10Apr.csv",
            "23Jun": "ahaan-code/results/saved_news/JPM_US_data_23Jun.csv",
        },
    },
    "INTC": {
        "company_name": "Intel",
        "aliases": {"INTC"},
        "data_paths": {"01May": "ahaan-code/results/saved_news/INTC_US_data_01May.csv"},
    },
    "TM": {
        "company_name": "Toyota",
        "aliases": {"TM"},
        "data_paths": {"06Mar": "ahaan-code/results/saved_news/TM_US_data_06Mar.csv"},
    },
    "SONY": {
        "company_name": "Sony Group",
        "aliases": {"SONY"},
        "data_paths": {"12Jun": "ahaan-code/results/saved_news/SONY_US_data_12Jun.csv"},
    },
    "DELL": {
        "company_name": "Dell Technologies",
        "aliases": {"DELL"},
        "data_paths": {"29May": "ahaan-code/results/saved_news/DELL_US_data_29May.csv"},
    },
}

ACTIVE_TICKER = "DELL"    # change this to any ticker key in TICKER_CONFIGS
ACTIVE_WEEK = "29May"     # change this to one of the week keys in that ticker's data_paths

if __name__ == "__main__":
    config = TICKER_CONFIGS[ACTIVE_TICKER]
    reviewer = FinBertReviewer(
        ticker=ACTIVE_TICKER,
        company_name=config["company_name"],
        aliases=config["aliases"],
    )
    data_path = config["data_paths"][ACTIVE_WEEK]
    output_path = f"ahaan-code/results/{ACTIVE_TICKER}_US_data_finbert_{ACTIVE_WEEK}.csv"

    rw_sentiment = reviewer.score(data_path, output_path)
    print(f"\nRelevance-weighted overall sentiment for {ACTIVE_TICKER} ({ACTIVE_WEEK}): {rw_sentiment:.4f}")
