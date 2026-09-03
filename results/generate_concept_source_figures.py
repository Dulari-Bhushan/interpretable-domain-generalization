"""Plan 07 (Stage 1 + Stage 2): bar charts comparing CLIP zero-shot vs.
DINOv2 linear-probe concept sources.

Usage: python generate_concept_source_figures.py
Reads results/concept_source_cub_stage1.json and
results/concept_source_cub_stage2_downstream.json, writes PNGs to
results/figures/.
"""
import json
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABELS = {
    "clip_vitl14_zeroshot": "CLIP ViT-L/14\n(zero-shot)",
    "dinov2_vitb14": "DINOv2 ViT-B/14\n(trained probe)",
    "dinov2_vitl14": "DINOv2 ViT-L/14\n(trained probe)",
}
COLORS = {
    "clip_vitl14_zeroshot": "#c0392b",
    "dinov2_vitb14": "#2471a3",
    "dinov2_vitl14": "#1a5276",
}
ORDER = ["clip_vitl14_zeroshot", "dinov2_vitb14", "dinov2_vitl14"]


def main():
    stage1 = json.load(open(os.path.join(RESULTS_DIR, "concept_source_cub_stage1.json")))
    stage2 = json.load(open(os.path.join(RESULTS_DIR, "concept_source_cub_stage2_downstream.json")))

    # --- Figure 1: Stage 1 - mean per-concept AUROC ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [stage1["variants"][k]["mean_auroc"] for k in ORDER]
    bars = ax.bar([LABELS[k] for k in ORDER], means, color=[COLORS[k] for k in ORDER])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="chance (AUROC=0.5)")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Mean per-concept AUROC\n(vs. real CUB 312-attribute labels, test split)")
    ax.set_title("Stage 1: concept-level agreement")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "concept_source_stage1_auroc.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: Stage 2 - downstream accuracy, in-domain + shift ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(ORDER))
    width = 0.35
    in_dom = [stage2["variants"][k]["acc_in_domain"] * 100 for k in ORDER]
    shift = [stage2["variants"][k]["acc_domain_shift"] * 100 for k in ORDER]
    bars1 = ax.bar([i - width / 2 for i in x], in_dom, width, label="In-domain (CUB test)", color="#7fb3d5")
    bars2 = ax.bar([i + width / 2 for i in x], shift, width, label="Domain shift (CUB-Painting)", color="#1a5276")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", fontsize=9)
    # key name in the actual JSON is "phase0_baseline_in_domain_acc" (a
    # labeling bug in the run that produced this file - the number itself is
    # Phase 0's CUB-Painting/domain-shift accuracy, not in-domain; see the
    # results writeup's Method section for the full explanation)
    phase0_shift = stage2.get("phase0_baseline_domain_shift_acc", stage2.get("phase0_baseline_in_domain_acc")) * 100
    ax.axhline(phase0_shift, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Phase 0 baseline, CUB-Painting ({phase0_shift:.2f}%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[k] for k in ORDER])
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("Stage 2: downstream accuracy (no DDO)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "concept_source_stage2_accuracy.png"), dpi=150)
    plt.close(fig)

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
