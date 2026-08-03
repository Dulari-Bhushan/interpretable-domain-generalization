"""Plot Phase F4's GenImage/Midjourney alignment result alongside Phase F1
(EuroSAT) and Phase F3 (Defactify) - a three-way comparison across two
methodologies (CLIP-text photo proxy vs. real matched photos).
"""
import json
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(REPO_ROOT, "results", "phase_f4_results.json")) as f:
    f4 = json.load(f)

lo, hi = f4["paper_alignment_range"]
bars = [
    ("EuroSAT (F1)\ntext-proxy photo\nmodality scarcity", f4["eurosat_f1_mean_alignment"], "#e67e22"),
    ("GenImage/Midjourney (F4)\ntext-proxy photo\ntemporal novelty", f4["mean_alignment"], "#9b59b6"),
    ("Defactify (F3)\nreal matched photos\ntemporal novelty", f4["defactify_f3_mean_alignment"], "#2980b9"),
]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.axhspan(lo, hi, color="#2ecc71", alpha=0.18, label=f"paper's own range ({lo}-{hi})")
labels = [b[0] for b in bars]
values = [b[1] for b in bars]
colors = [b[2] for b in bars]
ax.bar(labels, values, color=colors)
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Cosine similarity (visual shift vs. textual shift)")
ax.set_title("Pillar 2: alignment score across three datasets and two methodologies\n(text-proxy photo reference vs. real matched photos)")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_f4_three_way_comparison.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
