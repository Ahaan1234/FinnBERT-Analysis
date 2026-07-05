import os 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import csv 
import warnings
warnings.filterwarnings("ignore")


import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch.nn.functional as F

from config import TICKERS, VOLATILE_WEEKS, FINBERT_MODEL, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)



print(f"Loading FinBERT from '{FINBERT_MODEL}' …")
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
model.eval()
 

LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
 
 
def finbert_score(text: str) -> dict:
    """Return FinBERT probabilities and predicted label for a piece of text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    # Compute a composite score: positive_prob - negative_prob  ∈ [-1, +1]
    score = probs[0] - probs[1]
    return {
        "positive_prob": round(probs[0], 4),
        "negative_prob": round(probs[1], 4),
        "neutral_prob":  round(probs[2], 4),
        "predicted_label": LABEL_MAP[pred_idx],
        "sentiment_score": round(score, 4),   # composite score
    }
 
 
def build_prompt_sentences(ticker: str, company: str) -> list[str]:
    """
    Build minimal description sentences for a company.
    These are the only 'inputs' FinBERT sees in method 3.
    We use several phrasings and average the results.
    """
    return [
        f"The stock {ticker} belongs to {company}.",
        f"{company} is a publicly listed company traded under the ticker {ticker}.",
        f"News sentiment analysis for {company} ({ticker}).",
        f"Financial outlook for {ticker}, {company}.",
    ]
 
 
# ── Main analysis ──────────────────────────────────────────────────────────────
all_results = []
 
print("\n=== METHOD 3: FinBERT — No data (company name / ticker only) ===\n")
 
for ticker, company in TICKERS.items():
    sentences = build_prompt_sentences(ticker, company)
 
    # Score each sentence and average the composite scores
    sentence_scores = []
    agg_pos = agg_neg = agg_neu = 0.0
    for sent in sentences:
        s = finbert_score(sent)
        sentence_scores.append({
            "sentence": sent,
            **s,
        })
        agg_pos += s["positive_prob"]
        agg_neg += s["negative_prob"]
        agg_neu += s["neutral_prob"]
 
    n = len(sentences)
    avg_score  = round((agg_pos - agg_neg) / n, 4)
    avg_pos    = round(agg_pos / n, 4)
    avg_neg    = round(agg_neg / n, 4)
    avg_neu    = round(agg_neu / n, 4)
 
    # Derive label from averaged probs
    probs_avg  = [avg_pos, avg_neg, avg_neu]
    pred_label = LABEL_MAP[int(probs_avg.index(max(probs_avg)))]
 
    row = {
        "method":           "3_finbert_no_data",
        "ticker":           ticker,
        "company":          company,
        # Method 3 has no week-specific data — score is the same for both weeks
        "week":             "both (no data dependency)",
        "sentiment_score":  avg_score,
        "positive_prob":    avg_pos,
        "negative_prob":    avg_neg,
        "neutral_prob":     avg_neu,
        "predicted_label":  pred_label,
        "num_sentences":    n,
        "sentence_details": sentence_scores,
    }
    all_results.append(row)
 
    print(f"  {ticker:18s} ({company})")
    print(f"    label={pred_label:8s}  score={avg_score:+.4f}  "
          f"[pos={avg_pos:.3f} neg={avg_neg:.3f} neu={avg_neu:.3f}]")
 
# ── Save CSV (flat, without sentence_details) ──────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "method3_finbert_no_data.csv")
csv_fields = [
    "method", "ticker", "company", "week",
    "sentiment_score", "positive_prob", "negative_prob", "neutral_prob",
    "predicted_label", "num_sentences",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for row in all_results:
        writer.writerow({k: row[k] for k in csv_fields})
 
# ── Save full JSON (includes sentence-level detail) ────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "method3_finbert_no_data.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
 
print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
print("\nDone — Method 3 complete.") 