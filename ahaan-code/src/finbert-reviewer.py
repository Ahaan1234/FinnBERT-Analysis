# Score each news article for a ticker on two axes: FinBERT sentiment (bullish
# vs bearish) and relevance (how much the article is actually about the ticker,
# based on how crowded the symbol list is, whether the title mentions it, how
# densely the body text mentions it, and how recent it is). Combine both into
# a single relevance-weighted sentiment number for the ticker.

import ast
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

TICKER = "AAPL"
COMPANY_NAME = "Apple"
RECENCY_HALF_LIFE_DAYS = 7

TICKER_ALIASES = {
    "AAPL": {"AAPL", "APC"},
}

RELEVANCE_WEIGHTS = {
    "symbol_crowding": 0.4,
    "title_mention": 0.05,
    "content_density": 0.55,
    "recency": 0.1,
}

finbert_model_path = "ProsusAI/finbert"
sentiment_tokenizer = AutoTokenizer.from_pretrained(finbert_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_path)
sentiment_model.eval()
sentiment_labels = ["positive", "negative", "neutral"]

news_data_path = "ahaan-code/results/AAPL_US_data.csv"
articles_df = pd.read_csv(news_data_path)

mention_regex = re.compile(rf"\b({re.escape(TICKER)}|{re.escape(COMPANY_NAME)})\b", re.IGNORECASE)
aliases_for_target_ticker = TICKER_ALIASES.get(TICKER, {TICKER})

article_dates = pd.to_datetime(articles_df["date"], utc=True, errors="coerce")
most_recent_article_date = article_dates.max()

predicted_sentiment_labels = []
positive_scores = []
negative_scores = []
neutral_scores = []
sentiment_scores = []
symbol_crowding_scores = []
title_mention_scores = []
content_density_raw_scores = []
recency_scores = []

for row_index, article_row in articles_df.iterrows():
    article_text = article_row["content"]
    article_title = article_row["title"]
    article_symbols_text = article_row["symbols"]

    if isinstance(article_text, str) and article_text.strip():
        tokenized_article = sentiment_tokenizer(
            article_text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            model_output = sentiment_model(**tokenized_article)

        sentiment_probabilities = torch.nn.functional.softmax(model_output.logits, dim=-1)[0]
        label_to_score = {
            label: sentiment_probabilities[i].item() for i, label in enumerate(sentiment_labels)
        }
        predicted_label = max(label_to_score, key=label_to_score.get)

        predicted_sentiment_labels.append(predicted_label)
        positive_scores.append(label_to_score["positive"])
        negative_scores.append(label_to_score["negative"])
        neutral_scores.append(label_to_score["neutral"])
        sentiment_scores.append(label_to_score["positive"] - label_to_score["negative"])
    else:
        predicted_sentiment_labels.append(None)
        positive_scores.append(None)
        negative_scores.append(None)
        neutral_scores.append(None)
        sentiment_scores.append(None)

    try:
        symbols_on_article = ast.literal_eval(article_symbols_text)
    except (ValueError, SyntaxError):
        symbols_on_article = []

    other_company_roots = set()
    for symbol in symbols_on_article:
        symbol_root = symbol.split(".")[0].upper()
        symbol_root_without_digits = re.sub(r"\d+", "", symbol_root)
        normalized_symbol_root = symbol_root_without_digits if symbol_root_without_digits else symbol_root
        if normalized_symbol_root not in aliases_for_target_ticker:
            other_company_roots.add(normalized_symbol_root)
    other_company_count = len(other_company_roots)
    symbol_crowding_scores.append(1 / (1 + other_company_count))

    title_mentions_ticker = bool(mention_regex.search(article_title)) if isinstance(article_title, str) else False
    title_mention_scores.append(1.0 if title_mentions_ticker else 0.0)

    if isinstance(article_text, str) and article_text.strip():
        word_count_in_article = max(len(article_text.split()), 1)
        mention_hits_in_article = len(mention_regex.findall(article_text))
        content_density_raw_scores.append(mention_hits_in_article / word_count_in_article)
    else:
        content_density_raw_scores.append(0.0)

    article_date = article_dates[row_index]
    if pd.isna(article_date):
        recency_scores.append(0.0)
    else:
        article_age_in_days = (most_recent_article_date - article_date).total_seconds() / 86400
        recency_scores.append(0.5 ** (article_age_in_days / RECENCY_HALF_LIFE_DAYS))

lowest_content_density = min(content_density_raw_scores)
highest_content_density = max(content_density_raw_scores)
content_density_spread = highest_content_density - lowest_content_density

normalized_content_density_scores = []
for raw_density in content_density_raw_scores:
    if content_density_spread < 1e-9:
        normalized_content_density_scores.append(0.0)
    else:
        normalized_content_density_scores.append((raw_density - lowest_content_density) / content_density_spread)

relevance_scores = []
for i in range(len(articles_df)):
    relevance_scores.append(
        RELEVANCE_WEIGHTS["symbol_crowding"] * symbol_crowding_scores[i]
        + RELEVANCE_WEIGHTS["title_mention"] * title_mention_scores[i]
        + RELEVANCE_WEIGHTS["content_density"] * normalized_content_density_scores[i]
        + RELEVANCE_WEIGHTS["recency"] * recency_scores[i]
    )

articles_df["finbert_sentiment"] = predicted_sentiment_labels
articles_df["finbert_positive"] = positive_scores
articles_df["finbert_negative"] = negative_scores
articles_df["finbert_neutral"] = neutral_scores
articles_df["sentiment_score"] = sentiment_scores
articles_df["symbol_crowding_score"] = symbol_crowding_scores
articles_df["title_mention_score"] = title_mention_scores
articles_df["content_density_score"] = normalized_content_density_scores
articles_df["recency_score"] = recency_scores
articles_df["relevance_score"] = relevance_scores

total_weighted_sentiment = 0.0
total_relevance_weight = 0.0
for sentiment_value, relevance_value in zip(sentiment_scores, relevance_scores):
    total_weighted_sentiment += sentiment_value * relevance_value
    total_relevance_weight += relevance_value
relevance_weighted_overall_sentiment = total_weighted_sentiment / total_relevance_weight

output_path = "ahaan-code/results/AAPL_US_data_finbert.csv"
articles_df.to_csv(output_path, index=False)

print(articles_df[["date", "title", "finbert_sentiment", "sentiment_score", "relevance_score"]])
print(f"\nRelevance-weighted overall sentiment for {TICKER}: {relevance_weighted_overall_sentiment:.4f}")
