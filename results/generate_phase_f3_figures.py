"""Plot Phase F3's temporal-novelty alignment results (Defactify/MS-COCO-AI,
5 post-CLIP/post-GPT-3.5 generators) against Phase F1's EuroSAT (modality
scarcity) result and the paper's own claimed range.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(REPO_ROOT, "results", "phase_f3_results.json")) as f:
    f3 = json.load(f)

names = list(f3["global"]["per_generator"].keys())
release_dates = [f3["global"]["per_generator"][n]["release_date"] for n in names]
labels = [f"{n.replace(' generated image', '')}\n({d})" for n, d in zip(names, release_dates)]
global_vals = [f3["global"]["per_generator"][n]["alignment"] for n in names]
per_class_vals = [f3["per_class_controlled"]["per_generator"][n]["mean_alignment"] for n in names]

lo, hi = f3["paper_alignment_range"]
eurosat = f3["eurosat_f1_mean_alignment"]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axhspan(lo, hi, color="#2ecc71", alpha=0.18, label=f"paper's own range ({lo}-{hi})\nsketch/sculpture/painting shifts")
ax.axhline(eurosat, color="#e67e22", linestyle="--", linewidth=1.5,
           label=f"Phase F1 EuroSAT (modality scarcity): {eurosat}")

x = np.arange(len(names))
width = 0.35
ax.bar(x - width / 2, global_vals, width, label="global (mixed-content)", color="#7f8c8d")
ax.bar(x + width / 2, per_class_vals, width, label="per-class-controlled", color="#8e44ad")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Cosine similarity (visual shift vs. textual shift)")
ax.set_title("Phase F3: photo -> AI-generated alignment,\n5 generators released after both CLIP and GPT-3.5's cutoffs")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_f3_alignment_temporal.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
