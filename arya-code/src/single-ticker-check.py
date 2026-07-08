import sys
import yfinance as yf
import pandas as pd

TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
START = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
END = sys.argv[3] if len(sys.argv) > 3 else "2026-07-01"

data = yf.download(TICKER, start=START, end=END, progress=False)["Close"]
if isinstance(data, pd.DataFrame):
    data = data.squeeze("columns")
data = data.dropna()

weekly = data.resample("W-FRI").last().dropna()

weekly_returns = weekly.pct_change().dropna()

best_week = weekly_returns.idxmax()
worst_week = weekly_returns.idxmin()

print(f"=== Most volatile weeks for {TICKER} ===")
print(f"Max POSITIVE return week: {best_week.date()}  (return: {weekly_returns.loc[best_week]:.2%})")
print(f"Max NEGATIVE return week: {worst_week.date()}  (return: {weekly_returns.loc[worst_week]:.2%})")

out_path = f"{TICKER}_weekly_returns.csv"
weekly_returns.to_csv(out_path, header=["weekly_return"])
print(f"\nFull weekly return series saved to: {out_path}")