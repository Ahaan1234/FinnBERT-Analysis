import yfinance as yf
import pandas as pd

TICKERS = [
    # 3 randomly picked from the S&P 500 top 10 by market cap
    "JPM",
    "TSLA",
    "AVGO",
    # 4 big Indian stocks
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    # 3 big Japanese stocks
    "7203.T",   # Toyota
    "9984.T",   # SoftBank Group
    "6758.T",   # Sony
]
START = "2024-01-01"
END = "2026-07-01"

data = yf.download(TICKERS, start=START, end=END, progress=False)["Close"]
data = data.dropna(how="all")

weekly = data.resample("W-FRI").last()

weekly_returns = weekly.pct_change().dropna(how="all")

weekly_returns["basket_avg_return"] = weekly_returns.mean(axis=1)

best_week = weekly_returns["basket_avg_return"].idxmax()
worst_week = weekly_returns["basket_avg_return"].idxmin()

print("=== Basket-wide most volatile weeks ===")
print(f"Max POSITIVE return week: {best_week.date()}  "
      f"(avg return: {weekly_returns.loc[best_week, 'basket_avg_return']:.2%})")
print(f"Max NEGATIVE return week: {worst_week.date()}  "
      f"(avg return: {weekly_returns.loc[worst_week, 'basket_avg_return']:.2%})")

print("\nPer-ticker returns in the max-positive week:")
print(weekly_returns.loc[best_week].drop("basket_avg_return").sort_values(ascending=False))

print("\nPer-ticker returns in the max-negative week:")
print(weekly_returns.loc[worst_week].drop("basket_avg_return").sort_values())

weekly_returns.to_csv("basket_weekly_returns.csv")