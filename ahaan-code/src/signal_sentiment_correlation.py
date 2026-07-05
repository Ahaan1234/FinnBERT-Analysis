# Check whether any of the four relevance signals (symbol crowding, title
# mention, content density, recency) are secretly correlated with sentiment
# itself. If one of them is, that signal isn't just measuring "relevance" -
# it's nudging the weighted sentiment average in a direction on its own,
# which would bias the final relevance-weighted sentiment number.

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

finbert_scored_data_path = "ahaan-code/results/AAPL_US_data_finbert.csv"
articles_df = pd.read_csv(finbert_scored_data_path)

signal_columns = {
    "symbol_crowding_score": "Symbol crowding (1 / other companies tagged)",
    "title_mention_score": "Title mention (binary)",
    "content_density_score": "Content density (mentions / word count, normalized)",
    "recency_score": "Recency (half-life decay)",
}

print("Correlation of each relevance signal with FinBERT sentiment_score:\n")

correlation_results = []
for column_name, description in signal_columns.items():
    pearson_r, pearson_p = pearsonr(articles_df[column_name], articles_df["sentiment_score"])
    spearman_rho, spearman_p = spearmanr(articles_df[column_name], articles_df["sentiment_score"])

    correlation_results.append({
        "signal": column_name,
        "description": description,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    })

    print(f"{description}")
    print(f"    Pearson  r={pearson_r:+.3f}  (p={pearson_p:.4f})")
    print(f"    Spearman rho={spearman_rho:+.3f}  (p={spearman_p:.4f})\n")

results_df = pd.DataFrame(correlation_results)

# also check the combined relevance_score itself, since that's what actually
# gets used as the weight in the final weighted-sentiment calculation
combined_pearson_r, combined_pearson_p = pearsonr(articles_df["relevance_score"], articles_df["sentiment_score"])
print(f"Combined relevance_score vs sentiment_score: r={combined_pearson_r:+.3f} (p={combined_pearson_p:.4f})")

# --- bar chart of Pearson r for each signal, so it's easy to spot which
# signal (if any) is most entangled with sentiment ---

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["red" if abs(r) > 0.2 else "steelblue" for r in results_df["pearson_r"]]
ax.barh(results_df["description"], results_df["pearson_r"], color=bar_colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson correlation with sentiment_score")
ax.set_title("Relevance signals vs. sentiment (red = |r| > 0.2)", pad=12)
plt.tight_layout()

plot_path = "ahaan-code/results/signal_sentiment_correlation.png"
plt.savefig(plot_path, dpi=150)
print(f"\nSaved plot to {plot_path}")

output_path = "ahaan-code/results/signal_sentiment_correlation.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved correlation table to {output_path}")
