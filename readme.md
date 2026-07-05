# FinSent — Sentiment Analysis Benchmark (Methods 3, 4, 5)

## What this does

Compares three approaches to generating FinBERT sentiment scores for
10 global stocks across the two most volatile weeks identified from
basket_weekly_returns.csv:

| Week | Dates | Avg basket return |
|------|-------|-------------------|
| Max positive | 2024-08-12 to 2024-08-16 | +6.45% |
| Max negative | 2025-03-31 to 2025-04-04 | −9.72% |

**Stocks:** JPM, TSLA, AVGO (US) · RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS (India) · 7203.T, 9984.T, 6758.T (Japan)

## Files

```
config.py                      # All settings: tickers, weeks, API key
method3_finbert_no_data.py     # Method 3: FinBERT, no articles (baseline)
method4_finbert_alphavantage.py # Method 4: FinBERT on Alpha Vantage news
method5_finbert_multisource.py  # Method 5: FinBERT + multi-source + cosine filter
compare_results.py             # Merge and print comparison table
results/                       # All CSV and JSON outputs land here
```

## Setup

```bash
pip install transformers torch yfinance requests sentence-transformers
```

Put your Alpha Vantage key in `config.py`:
```python
ALPHA_VANTAGE_API_KEY = "YOUR_KEY_HERE"
```
Free key at https://alphavantage.co (25 requests/day — enough for 10 tickers × 2 weeks = 20 calls).

## Running

Run in order — each method is independent, compare_results needs all three done.

```bash
python method3_finbert_no_data.py
python method4_finbert_alphavantage.py
python method5_finbert_multisource.py
python compare_results.py
```

Total runtime: ~5–8 minutes (most time is FinBERT inference and API calls).

## Output files

| File | Contents |
|------|----------|
| `results/method3_finbert_no_data.csv` | Per-ticker baseline scores |
| `results/method4_finbert_alphavantage.csv` | Per-ticker×week scores from AV+FinBERT |
| `results/method5_finbert_multisource.csv` | Per-ticker×week scores from multi-source+cosine+FinBERT |
| `results/comparison_table.csv` | Wide table: all methods side-by-side |
| `results/comparison_summary.txt` | Human-readable comparison printout |
| `results/*.json` | Full article-level detail for each method |

## How the pre-filter works (Method 5)

```
Raw articles (Alpha Vantage + GDELT)
    │
    ▼  Step 1: Jaccard deduplication (threshold 0.7)
    │
    ▼  Step 2: Keyword filter (company name or ticker in headline/summary)
    │
    ▼  Step 3: Cosine similarity with all-MiniLM-L6-v2
               Query = "{company} stock market financial news {ticker}"
               Keep top-8 articles with cosine_sim ≥ 0.15
    │
    ▼  FinBERT → recency-weighted mean sentiment score
```

## What to look for in results

1. **Method 3 scores near 0** — confirms FinBERT has no prior bias toward any ticker.
2. **Method 4 negative scores for negative week** — AV news feed is picking up the right signal.
3. **Method 5 vs Method 4** — does cosine filtering improve consistency?
4. **M4_av_score vs M4_score** — does running FinBERT add value over AV's own keyword sentiment?
5. **India/Japan tickers in M4** — likely zero articles (AV doesn't cover them well); M5 should do better via GDELT.

## Notes

- Alpha Vantage free tier is 25 req/day. With 10 tickers × 2 weeks = 20 calls, you're fine.
- GDELT is free and has no rate limit, but results can be noisy — the cosine filter handles this.
- Indian (.NS) and Japanese (.T) tickers are not well covered by Alpha Vantage's news endpoint. Method 5 supplements with GDELT, which is multilingual and global.
- For the EY presentation, the key comparison is whether M5 (multi-source + pre-filtering) produces more directionally correct sentiment than M4 (single source, no filtering) during the volatile weeks. 