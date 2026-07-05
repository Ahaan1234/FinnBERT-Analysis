# Compare our hand-built relevance heuristic (see finbert-reviewer.py) against
# Arya's cosine-similarity approach, which embeds each article with
# all-MiniLM-L6-v2 and scores it against a query describing "relevant" news.
# Both scores run on the same AAPL article set to see where the methods agree/disagree.

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

TICKER = "AAPL"
COMPANY_NAME = "Apple"
TOP_N = 20

cosine_query_text = (
    f"{COMPANY_NAME} ({TICKER}) stock market financial news sentiment "
    f"earnings revenue guidance analyst price"
)

finbert_scored_data_path = "ahaan-code/results/AAPL_US_data_finbert.csv"
articles_df = pd.read_csv(finbert_scored_data_path)

print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2")

article_text_for_embedding = (articles_df["title"].fillna("") + ". " + articles_df["content"].fillna("")).str.strip()

cosine_query_embedding = sentence_embedder.encode(cosine_query_text, convert_to_tensor=True)
article_embeddings = sentence_embedder.encode(
    article_text_for_embedding.tolist(), convert_to_tensor=True, batch_size=32, show_progress_bar=True
)

cosine_similarity_scores = util.cos_sim(cosine_query_embedding, article_embeddings)[0].tolist()
articles_df["cosine_sim_score"] = cosine_similarity_scores

heuristic_relevance_scores = articles_df["relevance_score"]
cosine_relevance_scores = articles_df["cosine_sim_score"]

pearson_correlation, pearson_p_value = pearsonr(heuristic_relevance_scores, cosine_relevance_scores)
spearman_correlation, spearman_p_value = spearmanr(heuristic_relevance_scores, cosine_relevance_scores)

print(f"\nPearson correlation (linear agreement):  r={pearson_correlation:.3f}, p={pearson_p_value:.4f}")
print(f"Spearman correlation (rank agreement):    rho={spearman_correlation:.3f}, p={spearman_p_value:.4f}")

heuristic_top_n_indices = set(articles_df.nlargest(TOP_N, "relevance_score").index)
cosine_top_n_indices = set(articles_df.nlargest(TOP_N, "cosine_sim_score").index)
overlapping_top_n_indices = heuristic_top_n_indices & cosine_top_n_indices

print(f"\nTop-{TOP_N} overlap: {len(overlapping_top_n_indices)}/{TOP_N} articles picked by both methods")

articles_df["score_gap"] = (
    articles_df["relevance_score"].rank(pct=True) - articles_df["cosine_sim_score"].rank(pct=True)
)

print("\nArticles our heuristic rates far MORE relevant than cosine similarity does:")
print(articles_df.nlargest(5, "score_gap")[["title", "relevance_score", "cosine_sim_score"]].to_string(index=False))

print("\nArticles cosine similarity rates far MORE relevant than our heuristic does:")
print(articles_df.nsmallest(5, "score_gap")[["title", "relevance_score", "cosine_sim_score"]].to_string(index=False))

# top 10 articles where the two methods disagree the most, regardless of direction
articles_df["abs_score_gap"] = articles_df["score_gap"].abs()
top_10_disagreements = articles_df.nlargest(10, "abs_score_gap")[
    ["title", "relevance_score", "cosine_sim_score", "score_gap"]
]
print(f"\nTop 10 articles with the biggest disagreement between methods:")
print(top_10_disagreements.to_string(index=False))

# --- plots ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(articles_df["relevance_score"], bins=20, color="steelblue", edgecolor="black")
axes[0].set_title("Our heuristic relevance_score")
axes[0].set_xlabel("relevance_score")
axes[0].set_ylabel("number of articles")

axes[1].hist(articles_df["cosine_sim_score"], bins=20, color="darkorange", edgecolor="black")
axes[1].set_title("Cosine similarity score (MiniLM)")
axes[1].set_xlabel("cosine_sim_score")

plt.tight_layout()
distribution_plot_path = "ahaan-code/results/relevance_comparison_distributions.png"
plt.savefig(distribution_plot_path, dpi=150)
print(f"\nSaved distribution plot to {distribution_plot_path}")

# scatter plot of the two scores, with the top-10-disagreement articles highlighted
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(articles_df["relevance_score"], articles_df["cosine_sim_score"], alpha=0.4, label="all articles")
ax.scatter(
    top_10_disagreements["relevance_score"],
    top_10_disagreements["cosine_sim_score"],
    color="red",
    edgecolor="black",
    s=80,
    label="top 10 disagreements",
)
ax.set_xlabel("our heuristic relevance_score")
ax.set_ylabel("cosine_sim_score")
ax.set_title("Heuristic vs. cosine similarity relevance scores")
ax.legend()
plt.tight_layout()
scatter_plot_path = "ahaan-code/results/relevance_comparison_scatter.png"
plt.savefig(scatter_plot_path, dpi=150)
print(f"Saved scatter plot to {scatter_plot_path}")

comparison_output_path = "ahaan-code/results/relevance_comparison.csv"
articles_df[["date", "title", "relevance_score", "cosine_sim_score", "score_gap"]].to_csv(comparison_output_path, index=False)
print(f"\nSaved full comparison to {comparison_output_path}")
