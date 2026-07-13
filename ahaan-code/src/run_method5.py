# Runs arya-code's Config 5 (cosine-filtered FinBERT) over the tickers already
# scored by run.py, reusing the CSVs already saved under
# ahaan-code/results/saved_news/ instead of calling fetch_articles() again -
# no new API calls.

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "arya-code" / "src"))
import config
from method5_finbert_filtered import filter_articles, REFERENCE_QUERIES
from method4_finbert_articles import _score_articles

SAVED_NEWS_DIR = Path(__file__).resolve().parents[1] / "results" / "saved_news"

# ticker_display -> (saved CSV filename, reference query for cosine filtering)
TICKERS_TO_RUN = {
    "SFTBY":     ("SFTBY_US_data_26Jun.csv",     "SoftBank Group technology conglomerate telecom investment holding earnings"),
    "9988.HK":   ("9988.HK_US_data_15May.csv",   "Alibaba Group ecommerce cloud computing China retail earnings"),
    "MU":        ("MU_US_data_26Jun.csv",        "Micron Technology semiconductor memory chips DRAM NAND earnings"),
    "TM":        ("TM_US_data_08May.csv",        "Toyota Motor automobile car sales production EV hybrid Japan"),
    "HMC":       ("HMC_US_data_13May.csv",       "Honda Motor automobile motorcycle car sales production Japan"),
    "NOVO-B.CO": ("NOVO-B.CO_US_data_22Jun.csv", "Novo Nordisk pharmaceutical diabetes obesity drugs GLP-1 earnings"),
    "SAP.XETRA": ("SAP.XETRA_US_data_24Apr.csv", "SAP enterprise software cloud ERP business applications earnings"),
    "FDX":       ("FDX_US_data_23Jun.csv",       "FedEx logistics shipping package delivery freight earnings"),
    "9888.HK":   ("9888.HK_US_data_08Jul.csv",   "Baidu search engine AI China internet earnings"),
    "SONY":      ("SONY_US_data_13May.csv",      "Sony Group electronics gaming PlayStation music entertainment Japan"),
}


def load_articles(csv_path):
    """Reads a saved_news CSV and converts rows into method5's article dict shape."""
    df = pd.read_csv(csv_path)
    articles = []
    for _, row in df.iterrows():
        title = row.get("title", "")
        content = row.get("content", "")
        title = title if isinstance(title, str) else ""
        content = content if isinstance(content, str) and content.strip() else title
        articles.append({"title": title, "content": content})
    return articles


def run_config5_from_cache():
    results = {}
    for ticker, (fname, ref_query) in TICKERS_TO_RUN.items():
        csv_path = SAVED_NEWS_DIR / fname
        if not csv_path.exists():
            print(f"  {ticker:<12} MISSING CSV ({fname}) — skipping")
            results[ticker] = None
            continue

        articles = load_articles(csv_path)
        n_raw = len(articles)
        if n_raw == 0:
            print(f"  {ticker:<12} NO ARTICLES — score=None")
            results[ticker] = {"score": None, "n_raw": 0, "n_filtered": 0, "fallback_used": False, "mean_sim": 0.0}
            continue

        REFERENCE_QUERIES[ticker] = ref_query
        filtered, sims = filter_articles(ticker, articles)
        n_filtered = len(filtered)
        fallback_used = False

        if n_filtered == 0:
            print(f"  {ticker:<12} cosine filter removed all articles "
                  f"(threshold={config.COSINE_THRESHOLD}) — using unfiltered fallback")
            filtered = articles
            sims = [0.0] * len(articles)
            n_filtered = len(filtered)
            fallback_used = True

        per_article_scores = _score_articles(filtered)
        mean_score = sum(per_article_scores) / len(per_article_scores)
        mean_sim = sum(sims) / len(sims) if sims else 0.0

        marker = " [FALLBACK]" if fallback_used else ""
        print(
            f"  {ticker:<12} raw={n_raw:>4}  kept={n_filtered:>4}  "
            f"mean_sim={mean_sim:.3f}  score={mean_score:+.4f}{marker}"
        )

        results[ticker] = {
            "score": round(mean_score, 4),
            "n_raw": n_raw,
            "n_filtered": n_filtered,
            "fallback_used": fallback_used,
            "mean_sim": round(mean_sim, 4),
        }

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("CONFIG 5 (cosine-filtered FinBERT) — reusing cached ahaan-code CSVs")
    print("=" * 80)
    results = run_config5_from_cache()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for t, r in results.items():
        if r is None or r["score"] is None:
            print(f"  {t:<12} N/A")
        else:
            fb = " *fallback*" if r["fallback_used"] else ""
            print(
                f"  {t:<12} {r['score']:+.4f}  "
                f"(raw={r['n_raw']}, kept={r['n_filtered']}, mean_sim={r['mean_sim']:.3f}){fb}"
            )
