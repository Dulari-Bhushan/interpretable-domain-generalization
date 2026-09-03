"""Plan 08 extended to PACS: bar chart, human vs. LLM concept bank, both
alpha values, all 4 domains (photo=in-domain, art_painting/cartoon/sketch=
shift).

Usage: python generate_pacs_concept_bank_figures.py
Reads results/pacs_concept_bank_comparison.json, writes a PNG to
results/figures/.
"""
import json
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
DOMAIN_LABELS = ["photo\n(in-domain)", "art_painting\n(shift)", "cartoon\n(shift)", "sketch\n(shift)"]


def main():
    data = json.load(open(os.path.join(RESULTS_DIR, "pacs_concept_bank_comparison.json")))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, alpha, title in zip(axes, ["0.0", "1.0"], ["No DDO (alpha=0)", "+DDO (alpha=1)"]):
        human = [data[f"human_alpha{alpha}"][d] * 100 for d in DOMAINS]
        llm = [data[f"llm_alpha{alpha}"][d] * 100 for d in DOMAINS]
        x = range(len(DOMAINS))
        width = 0.35
        bars1 = ax.bar([i - width / 2 for i in x], human, width, label="Human bank", color="#2471a3")
        bars2 = ax.bar([i + width / 2 for i in x], llm, width, label="LLM bank", color="#c0392b")
        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(DOMAIN_LABELS, fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Classification accuracy (%)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("PACS: human vs. LLM-generated concept bank (trained on photo)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pacs_concept_bank_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
