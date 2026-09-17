"""Plan 09: figures for the VLM/DINO+CLIP concept-generation comparison.

Usage: python generate_vlm_concept_generation_figures.py
Numbers sourced directly from:
  external/LanCE/data/CUB/vlm_concepts/quality_eval_results.json (size_normalized blocks)
  results/plan09_downstream_accuracy_comparison.json (banks + size_controlled_followup blocks)
Writes two PNGs to results/figures/.
"""
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Figure 1: size-normalized (n=263) match to LanCE's own concept bank ---
QUALITY = {
    "Human\n(LanCE, n=312)": (1.0, 1.0),  # reference bank matched against itself, shown for scale only
    "LLM draw 1\n(n=312)": (0.8991, 0.9040),
    "LLM draw 2\n(n=312)": (0.8143, 0.8181),
    "Qwen2-VL\n(n=263)": (0.7692, 0.7814),
    "Gemini\n(n=1842)": (0.8028, 0.8054),
    "DINO+CLIP\n(n=1328)": (0.8259, 0.8104),
}


def fig_quality():
    labels = list(QUALITY.keys())
    coverage = [QUALITY[k][0] for k in labels]
    precision = [QUALITY[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], coverage, width, label="Coverage", color="#2471a3")
    ax.bar([i + width / 2 for i in x], precision, width, label="Precision", color="#c0392b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean best-match CLIP similarity to LanCE's bank")
    ax.set_title("Match to LanCE's own concept bank, size-normalized to 263 concepts\n(human bank shown at n=312, unnormalized, for scale)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vlm_concept_match_lance.png"), dpi=150)
    plt.close(fig)


# --- Figure 2: downstream accuracy, full-size vs size-controlled ---
# (target/CUB-Painting accuracy, alpha=1/+DDO, the project's headline metric)
FIXED = {
    "Human\n(n=312)": 57.04,
    "LLM draw 1\n(n=312)": 59.07,
    "LLM draw 2\n(n=312)": 61.40,
    "Qwen2-VL\n(n=263)": 57.63,
}
CONFOUNDED = {
    "Gemini": (63.44, 58.65, 1842),
    "DINO+CLIP": (63.93, 60.16, 1328),
}


def fig_downstream():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    fixed_labels = list(FIXED.keys())
    fixed_vals = [FIXED[k] for k in fixed_labels]
    x_fixed = range(len(fixed_labels))
    ax.bar(x_fixed, fixed_vals, width=0.5, color="#7f8c8d", label="Fixed-size baselines")
    for i, v in zip(x_fixed, fixed_vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)

    offset = len(fixed_labels)
    conf_labels = list(CONFOUNDED.keys())
    for j, name in enumerate(conf_labels):
        full_acc, controlled_acc, full_n = CONFOUNDED[name]
        xi = offset + j * 2.4
        b1 = ax.bar(xi, full_acc, width=0.5, color="#c0392b", label="Full size (confounded)" if j == 0 else None)
        b2 = ax.bar(xi + 0.6, controlled_acc, width=0.5, color="#2471a3", label="Size-controlled (n=263)" if j == 0 else None)
        ax.text(xi, full_acc + 0.8, f"{full_acc:.1f}%\n(n={full_n})", ha="center", fontsize=8.5)
        ax.text(xi + 0.6, controlled_acc + 0.8, f"{controlled_acc:.1f}%", ha="center", fontsize=8.5)
        ax.text(xi + 0.3, -4.5, name, ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(list(x_fixed))
    ax.set_xticklabels(fixed_labels, fontsize=9)
    ax.set_ylabel("CUB-Painting (shift) accuracy, alpha=1/+DDO")
    ax.set_title("Downstream accuracy: the full-size Gemini/DINO+CLIP lead is mostly a bank-size effect")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vlm_concept_downstream_accuracy.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_quality()
    fig_downstream()
    print(f"Wrote figures to {FIG_DIR}")
