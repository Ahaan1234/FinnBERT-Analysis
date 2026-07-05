# given a ticker, call on the EODHD API to receive a json. 
# Store this json to avoid overusing limit free api calls xP
# from "content" print the first 5 words of each summary. 

import requests
import pandas as pd

API_TOKEN = "demo"
ticker = "AAPL.US"

params = {
    "s": ticker,
    "from": "2024-08-01",
    "to": "2025-07-28",
    "limit":100,
    "api_token": API_TOKEN,
    "fmt": "json"
}

r = requests.get("https://eodhd.com/api/news", params=params, timeout=30)
r.raise_for_status()
articles = r.json()

df = pd.json_normalize(articles)
df.to_csv(f'ahaan-code/{ticker.replace(".","_")}_data.csv', index=False)

count = 1
for article in articles:
    print(f"{count}. title: ", article["title"], "\nexcerpt: ", article["content"][:30])
    count+=1

