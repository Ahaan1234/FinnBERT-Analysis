# Runs arya-code's Config 6 (FAISS bi-encoder recall -> cross-encoder rerank
# -> NER boost -> FinBERT) over the tickers already scored by run.py, reusing
# the CSVs already saved under ahaan-code/results/saved_news/ instead of
# calling fetch_articles() again - no new API calls.
#
# Our 10 tickers aren't in arya-code's ENTITY_CATALOGUE (that's a fixed
# 10-stock basket from a different project), so the NER stage just scores
# them 0 - fine, since that stage is additive and everything else (FAISS
# recall + cross-encoder rerank + FinBERT) still runs normally as long as
# we give it a reference query per ticker.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # faiss + torch OpenMP conflict causes a segfault otherwise
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "arya-code" / "src"))
import config
import faiss_index
from semantic_relevance import filter_articles_semantic
from method3_finbert_no_data import get_finbert, _scores_to_scalar

SAVED_NEWS_DIR = Path(__file__).resolve().parents[1] / "results" / "saved_news"

# ticker_display -> (saved CSV filename, date window, reference query)
TICKERS_TO_RUN = {
    "SFTBY":     ("SFTBY_US_data_26Jun.csv",     {"label": "17-26 Jun 2026", "start": "2026-06-17", "end": "2026-06-26"}, "SoftBank Group technology conglomerate telecom investment holding earnings"),
    "9988.HK":   ("9988.HK_US_data_15May.csv",   {"label": "6-15 May 2026",  "start": "2026-05-06", "end": "2026-05-15"}, "Alibaba Group ecommerce cloud computing China retail earnings"),
    "MU":        ("MU_US_data_26Jun.csv",        {"label": "17-26 Jun 2026", "start": "2026-06-17", "end": "2026-06-26"}, "Micron Technology semiconductor memory chips DRAM NAND earnings"),
    "TM":        ("TM_US_data_08May.csv",        {"label": "1-8 May 2026",   "start": "2026-05-01", "end": "2026-05-08"}, "Toyota Motor automobile car sales production EV hybrid Japan"),
    "HMC":       ("HMC_US_data_13May.csv",       {"label": "6-13 May 2026",  "start": "2026-05-06", "end": "2026-05-13"}, "Honda Motor automobile motorcycle car sales production Japan"),
    "NOVO-B.CO": ("NOVO-B.CO_US_data_22Jun.csv", {"label": "15-22 Jun 2026", "start": "2026-06-15", "end": "2026-06-22"}, "Novo Nordisk pharmaceutical diabetes obesity drugs GLP-1 earnings"),
    "SAP.XETRA": ("SAP.XETRA_US_data_24Apr.csv", {"label": "17-24 Apr 2026", "start": "2026-04-17", "end": "2026-04-24"}, "SAP enterprise software cloud ERP business applications earnings"),
    "FDX":       ("FDX_US_data_23Jun.csv",       {"label": "16-23 Jun 2026", "start": "2026-06-16", "end": "2026-06-23"}, "FedEx logistics shipping package delivery freight earnings"),
    "9888.HK":   ("9888.HK_US_data_08Jul.csv",   {"label": "1-8 Jul 2026",   "start": "2026-07-01", "end": "2026-07-08"}, "Baidu search engine AI China internet earnings"),
    "SONY":      ("SONY_US_data_13May.csv",      {"label": "6-13 May 2026",  "start": "2026-05-06", "end": "2026-05-13"}, "Sony Group electronics gaming PlayStation music entertainment Japan"),
}

# Register our reference queries so both the FAISS recall stage and the
# cross-encoder reranker (both import this same dict) know what to match on.
for ticker, (_, _, ref_query) in TICKERS_TO_RUN.items():
    faiss_index.TICKER_REFERENCE_QUERIES[ticker] = ref_query


def load_articles(csv_path):
    df = pd.read_csv(csv_path)
    articles = []
    for _, row in df.iterrows():
        title = row.get("title", "")
        content = row.get("content", "")
        title = title if isinstance(title, str) else ""
        content = content if isinstance(content, str) and content.strip() else title
        articles.append({"title": title, "content": content})
    return articles


def _score_articles(articles):
    clf = get_finbert()
    char_limit = config.FINBERT_MAX_LENGTH * 4
    texts = [f"{a['title']}. {a['content']}"[:char_limit] for a in articles]
    scores = []
    bs = config.FINBERT_BATCH_SIZE
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        results = clf(batch, truncation=True, max_length=config.FINBERT_MAX_LENGTH)
        for result in results:
            if isinstance(result, list) and isinstance(result[0], dict):
                scores.append(_scores_to_scalar(result))
            elif isinstance(result, list) and isinstance(result[0], list):
                scores.append(_scores_to_scalar(result[0]))
            else:
                scores.append(0.0)
    return scores


def run_config6_from_cache(top_k=100, top_m=20):
    results = {}
    for ticker, (fname, window, _) in TICKERS_TO_RUN.items():
        csv_path = SAVED_NEWS_DIR / fname
        if not csv_path.exists():
            print(f"  {ticker:<12} MISSING CSV ({fname}) — skipping")
            results[ticker] = None
            continue

        articles = load_articles(csv_path)
        n_raw = len(articles)
        if n_raw == 0:
            print(f"  {ticker:<12} NO ARTICLES — score=None")
            results[ticker] = {"score": None, "n_raw": 0, "n_filtered": 0, "window": window["label"]}
            continue

        filtered, meta = filter_articles_semantic(ticker, articles, window, top_k=top_k, top_m=top_m)
        n_filtered = len(filtered)

        if n_filtered == 0:
            print(f"  {ticker:<12} semantic filter returned 0 — score=None")
            results[ticker] = {"score": None, "n_raw": n_raw, "n_filtered": 0, "window": window["label"]}
            continue

        per_article_scores = _score_articles(filtered)
        mean_score = sum(per_article_scores) / len(per_article_scores)

        fb_marker = " [FALLBACK]" if meta["fallback_used"] else ""
        print(
            f"  {ticker:<12} raw={n_raw:>4}  recall={meta['n_recall']:>4}  "
            f"reranked={n_filtered:>3}  final_sim={meta['mean_final']:.3f}  "
            f"score={mean_score:+.4f}{fb_marker}"
        )

        results[ticker] = {
            "score": round(mean_score, 4),
            "n_raw": n_raw,
            "n_recall": meta["n_recall"],
            "n_filtered": n_filtered,
            "mean_final": meta["mean_final"],
            "fallback_used": meta["fallback_used"],
            "window": window["label"],
        }

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("CONFIG 6 (FAISS recall -> cross-encoder rerank -> NER -> FinBERT)")
    print("reusing cached ahaan-code CSVs — no new API calls")
    print("=" * 80)
    results = run_config6_from_cache()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for t, r in results.items():
        if r is None or r["score"] is None:
            print(f"  {t:<12} N/A")
        else:
            fb = " *fallback*" if r.get("fallback_used") else ""
            print(
                f"  {t:<12} {r['score']:+.4f}  "
                f"(raw={r['n_raw']}, recall={r.get('n_recall','-')}, kept={r['n_filtered']}){fb}"
            )
