import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Publication style
# -----------------------------
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300
})


loss_data = {

    "Mistral": {
        "epoch": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "loss":  [2.9574, 0.6645, 0.6872, 0.4299, 0.4470, 0.4404, 0.4316]
    }
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6.5, 4.5))

line_styles = ["-"]
markers = ["o"]

for (model, values), ls, mk in zip(loss_data.items(), line_styles, markers):
    plt.plot(
        values["epoch"],
        values["loss"],
        linestyle=ls,
        marker=mk,
        linewidth=2,
        markersize=5,
        label=model
    )

# Labels
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Loss Curve")

# Legend
plt.legend(frameon=True)

# Clean layout
plt.tight_layout()

# Save for paper
plt.savefig("loss_curve.pdf", bbox_inches="tight")
plt.savefig("loss_curve.png", dpi=300)

plt.show()
