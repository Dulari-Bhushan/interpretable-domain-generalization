"""Plot Phase A's descriptor-coverage dose-response from results/phase_a_results.json."""
import json
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(REPO_ROOT, "results", "phase_a_results.json")) as f:
    data = json.load(f)

conditions = data["conditions"]
mean_sims = [c["mean_similarity"] for c in conditions]
accs = [c["target_acc"] for c in conditions]
pool_sizes = [c["pool_size"] for c in conditions]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(mean_sims, accs, "o-", color="#2471a3", linewidth=2, markersize=7)
for sim, acc, n in zip(mean_sims, accs, pool_sizes):
    ax.annotate(f"n={n}", (sim, acc), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
ax.axhline(57.04, color="#7f8c8d", linestyle="--", alpha=0.6, linewidth=1, label="Phase 0 full-pool reference (57.04%)")
ax.set_xlabel("Mean cosine similarity of remaining descriptor pool to true 'painting' domain")
ax.set_ylabel("Target (CUB-Painting) accuracy (%)")
ax.set_title("Phase A: does DDO need a well-matched descriptor?")
ax.set_ylim(50, 62)
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "phase_a_descriptor_coverage.png"), dpi=150)
plt.close(fig)
print("Wrote", os.path.join(FIG_DIR, "phase_a_descriptor_coverage.png"))
