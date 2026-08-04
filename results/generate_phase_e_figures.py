"""Generate figures for Phase E2 (real photo -> Midjourney v6 DDO training run)
from results/phase_e2_results.json. Style matches generate_phase0_figures.py
(same color scheme, dpi, grid) since this is a direct comparison to Phase 0.

Usage: python generate_phase_e_figures.py
"""
import os
import json
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_COLOR = "#c0392b"
DDO_COLOR = "#2471a3"


def main():
    with open(os.path.join(RESULTS_DIR, "phase_e2_results.json")) as f:
        r = json.load(f)

    baseline = r["results_by_condition"]["baseline"]
    ddo = r["results_by_condition"]["ddo"]
    epochs = list(range(1, len(baseline["target_curve"]) + 1))

    # --- Figure 1: target (OOD, Midjourney v6) accuracy curves ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, baseline["target_curve"], label="baseline (α=0)", color=BASELINE_COLOR, linewidth=2)
    ax.plot(epochs, ddo["target_curve"], label="+DDO (α=1)", color=DDO_COLOR, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Target (Midjourney v6) accuracy (%)")
    ax.set_title("Phase E2: real photo → Midjourney v6, unmodified 204-descriptor pool")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase_e2_target_accuracy.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: DDO's gain, Phase 0 (descriptor pool covers "painting")
    # vs Phase E2 (descriptor pool covers nothing for this domain) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Phase 0\nCUB → Painting\n(pool covers this domain)", "Phase E2\nPhoto → Midjourney v6\n(pool covers nothing)"]
    gains = [r["phase0_ddo_gain_points_reference"], r["ddo_gain_points"]]
    bar_colors = [DDO_COLOR, "#7f8c8d"]
    bars = ax.bar(labels, gains, color=bar_colors, width=0.55)
    for bar, val in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"+{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("DDO gain over baseline, OOD target accuracy (points)")
    ax.set_title("DDO's benefit shrinks when its descriptor pool\nhas no term for the target domain")
    ax.set_ylim(0, max(gains) * 1.25)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase_e2_ddo_gain_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Figure 3: source (in-domain) vs target (Midjourney) accuracy, both
    # conditions - shows target is NOT depressed relative to source, despite
    # Phase F3's near-zero alignment score for this exact domain shift ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    conditions = ["baseline (α=0)", "+DDO (α=1)"]
    source_accs = [baseline["final_source_acc"], ddo["final_source_acc"]]
    target_accs = [baseline["final_target_acc"], ddo["final_target_acc"]]
    x = range(len(conditions))
    width = 0.32
    ax.bar([i - width / 2 for i in x], source_accs, width, label="Source test (real photos, in-domain)", color="#5d6d7e")
    ax.bar([i + width / 2 for i in x], target_accs, width, label="Target test (Midjourney v6, OOD)", color=DDO_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Trained accuracy: real photos vs. Midjourney v6\n(target is not depressed despite alignment ≈ 0.037, Phase F3)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase_e2_source_vs_target.png"), dpi=150)
    plt.close(fig)

    print(f"Baseline final target acc: {baseline['final_target_acc']:.2f}%")
    print(f"DDO final target acc: {ddo['final_target_acc']:.2f}%")
    print(f"DDO gain: {r['ddo_gain_points']:+.2f} points (Phase 0 reference: +{r['phase0_ddo_gain_points_reference']:.2f})")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
