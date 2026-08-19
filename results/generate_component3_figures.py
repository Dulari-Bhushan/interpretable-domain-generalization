"""Plots for Component 3's report (component3_self_growing_vocabulary.md) -
the candidate-scores-vs-threshold chart (why growth accepted 0/10) and the
4-condition target-accuracy comparison (grown == text-only, since nothing
was accepted). Pure plotting of numbers already in
results/component3_defactify_growing_vocab.json - no new runs.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def plot_candidate_scores():
    r = load("component3_defactify_growing_vocab.json")
    threshold = r["diagnosis"]["threshold"]
    rejected = r["growth"]["rejected"]
    accepted = r["growth"]["accepted"]
    items = sorted(list(accepted.items()) + list(rejected.items()), key=lambda kv: -kv[1])
    labels = [k.replace(" Midjourney ", "\nMidjourney ").replace(" of a {}.", "") for k, _ in items]
    values = [v for _, v in items]
    colors = ["#2ecc71" if k in accepted else "#e74c3c" for k, _ in items]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.6)
    ax.axhline(threshold, color="#34495e", linestyle="--", linewidth=1.5,
               label=f"calibrated trust threshold ({threshold:.3f})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel("Alignment score\n(cosine: real visual shift vs. candidate-template-predicted shift)")
    ax.set_title("Component 3 growth: all 10 auto-generated candidates for\n"
                 "photo→Midjourney v6, none clearing the trust threshold\n(green = accepted, red = rejected)")
    ax.legend(fontsize=9)
    for xi, v in zip(x, values):
        ax.text(xi, v + (0.003 if v >= 0 else -0.008), f"{v:.3f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "component3_candidate_scores.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


def plot_condition_comparison():
    r = load("component3_defactify_growing_vocab.json")
    conds = r["results_by_condition"]
    order = ["baseline", "ddo_text", "ddo_grounded", "ddo_grown"]
    labels = ["Baseline\n(α=0)", "+DDO\ntext-only\n(204 dir.)", "+DDO\ngrounded\n(Component 2)", "+DDO\ngrown\n(204+0 new)"]
    accs = [conds[k]["best_target_acc"] for k in order]
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#3498db"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, accs, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best target (Midjourney v6) accuracy (%)")
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    ax.set_title("Component 3: grown pool = text pool exactly\n(growth accepted 0/10 candidates, see other figure)")
    for xi, a in zip(x, accs):
        ax.text(xi, a + 0.1, f"{a:.2f}%", ha="center", fontsize=9)
    ax.annotate("identical\n(both 204-dir.,\nsame numbers)", xy=(1.5, max(accs[1], accs[3]) + 0.6),
                ha="center", fontsize=8, color="#2c3e50")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "component3_condition_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    plot_candidate_scores()
    plot_condition_comparison()
