"""Plan 08: bar chart comparing human-written vs. LLM-generated CUB concept
banks, baseline and +DDO, in-domain and CUB-Painting shift.

Usage: python generate_llm_concept_bank_figures.py
Reads the numbers directly (see NUMBERS below - sourced from
external/LanCE/logs_baseline_cached.log, logs_ddo_cached.log,
logs_plan08_llm_alpha0_cached.log, logs_plan08_llm_alpha1_cached.log's own
"Best Target Accuracy"/"Best Source Accuracy" lines), writes a PNG to
results/figures/.
"""
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# (source/in-domain %, target/CUB-Painting %), from each run's own
# "Training Completed" summary block (source_acc recorded at the same
# epoch as best target_acc, matching this project's own convention).
NUMBERS = {
    "Human bank\nbaseline": (77.81, 50.64),
    "LLM bank\nbaseline": (79.43, 53.82),
    "Human bank\n+DDO": (79.52, 57.04),
    "LLM bank\n+DDO": (81.24, 59.07),
}


def main():
    labels = list(NUMBERS.keys())
    source = [NUMBERS[k][0] for k in labels]
    target = [NUMBERS[k][1] for k in labels]
    colors_src = ["#aed6f1", "#7fb3d5", "#aed6f1", "#7fb3d5"]
    colors_tgt = ["#f5b7b1", "#c0392b", "#f5b7b1", "#c0392b"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(labels))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], source, width, label="In-domain (CUB test)", color="#2471a3")
    bars2 = ax.bar([i + width / 2 for i in x], target, width, label="Domain shift (CUB-Painting)", color="#c0392b")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", fontsize=9)
    ax.axvline(1.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("Plan 08: human-written vs. LLM-generated CUB concept bank")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "llm_concept_bank_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
