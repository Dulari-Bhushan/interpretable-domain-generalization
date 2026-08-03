"""Plot Pillar 2's EuroSAT results: Phase F1 (domain-shift alignment score)
and Phase F2 (concept-activation ceiling test).

Two figures:
  1. phase_f1_alignment_per_class.png - per-class alignment score bar chart,
     with the paper's own 0.90-0.99 range shaded for direct comparison.
  2. phase_f2_ceiling_curve.png - CBM test-accuracy curve over training,
     with zero-shot and linear-probe reference lines.
"""
import json
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(REPO_ROOT, "results", "phase_f1_results.json")) as f:
    f1 = json.load(f)
with open(os.path.join(REPO_ROOT, "results", "phase_f2_results.json")) as f:
    f2 = json.load(f)

# --- 1. Per-class alignment ---
fig, ax = plt.subplots(figsize=(8, 5))
classes = list(f1["per_class_alignment"].keys())
values = list(f1["per_class_alignment"].values())
lo, hi = f1["paper_alignment_range"]
ax.axhspan(lo, hi, color="#2ecc71", alpha=0.2, label=f"paper's own range ({lo}-{hi})\nsketch/sculpture/painting shifts")
ax.bar(classes, values, color="#e67e22")
ax.axhline(f1["mean_alignment"], color="#c0392b", linestyle="--", linewidth=1.5,
           label=f"mean alignment ({f1['mean_alignment']:.2f})")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Cosine similarity (visual shift vs. textual shift)")
ax.set_title("Phase F1: photo -> satellite domain-shift alignment, per EuroSAT class")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_f1_alignment_per_class.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

# --- 2. CBM ceiling curve ---
fig, ax = plt.subplots(figsize=(8, 5))
xs = list(range(1, len(f2["acc_curve"]) + 1))
ax.plot(xs, f2["acc_curve"], color="#2471a3", linewidth=2, label="trained CBM (alpha=0, no DDO)")
ax.axhline(f2["anchors"]["paper_linear_probe_acc_336px"], color="#27ae60", linestyle="--",
           label=f"linear-probe ceiling ({f2['anchors']['paper_linear_probe_acc_336px']}%, OpenAI, ViT-L/14-336px)")
ax.axhline(f2["anchors"]["our_zeroshot_acc"], color="#c0392b", linestyle="--",
           label=f"our zero-shot ({f2['anchors']['our_zeroshot_acc']:.1f}%, ViT-L/14)")
ax.axhline(f2["anchors"]["paper_zeroshot_acc_336px"], color="#e67e22", linestyle=":",
           label=f"paper's zero-shot ({f2['anchors']['paper_zeroshot_acc_336px']}%, ViT-L/14-336px)")
ax.set_xlabel("Training epoch")
ax.set_ylabel("EuroSAT test accuracy (%)")
ax.set_ylim(0, 102)
ax.set_title("Phase F2: does a trained concept-bottleneck classifier\ntrack the zero-shot ceiling or the linear-probe ceiling?")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_f2_ceiling_curve.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
