import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------
# Data
# -----------------------------
data = {
    "Method": ["Vanilla", "RAG+Vanilla", "Fine tuned", "RAG+Finetune"],
    "BLEU-4": [0.0274, 0.0515, 0.0561, 0.0451],
    "ROUGE-L": [0.1620, 0.2269, 0.1795, 0.1534],
    "BERTScore F1": [0.8728, 0.8900, 0.8356, 0.8297],
}

df = pd.DataFrame(data)

# Convert to long format (better for seaborn)
df_long = df.melt(id_vars="Method", 
                  var_name="Metric", 
                  value_name="Score")

# -----------------------------
# Style (modern + attractive)
# -----------------------------
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.3
)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=df_long,
    x="Method",
    y="Score",
    hue="Metric",
    palette="viridis"   # attractive color palette
)

# Improve appearance
plt.xticks(rotation=15)
plt.title("Comparison of Evaluation Metrics Across Methods", weight="bold")
plt.xlabel("Methods")
plt.ylabel("Score")
plt.legend(title="Metric", frameon=True)
sns.despine()

plt.tight_layout()

# -----------------------------
# Save high-quality figures
# -----------------------------
plt.savefig("metrics_comparison.png", dpi=600)
plt.savefig("metrics_comparison.pdf")

plt.show()