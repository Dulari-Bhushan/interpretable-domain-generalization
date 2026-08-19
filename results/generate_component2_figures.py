"""Plots for Component 2's two reports (component2_self_diagnosing_grounding.md,
component2b_grounding_fallback_variants.md) - the calibration curve (PACS/
EuroSAT/Defactify alignment scores vs. the fallback threshold) and the
fallback-variants comparison (probe-size sweep + persample + blend vs.
baseline/text-only DDO). Pure plotting of numbers already in the cited
results/*.json files - no new runs.
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


def plot_calibration_curve():
    pacs = load("component2_alignment_calibration.json")
    eurosat = load("component2_eurosat_calibration.json")
    defactify_mean = pacs["phase_f3_mean_alignment_cited"]
    threshold = pacs["calibrated_threshold"]

    labels, values, colors = [], [], []
    for domain, res in pacs["results_per_target_domain"].items():
        labels.append(f"photo→{domain}\n(PACS)")
        values.append(res["mean_alignment"])
        colors.append("#2ecc71")
    labels.append("photo→satellite\n(EuroSAT)")
    values.append(eurosat["mean_alignment"])
    colors.append("#27ae60")
    labels.append("photo→Midjourney v6\n(Defactify)")
    values.append(defactify_mean)
    colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.6)
    ax.axhline(threshold, color="#34495e", linestyle="--", linewidth=1.5,
               label=f"calibrated threshold ({threshold:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean alignment score\n(cosine: real visual shift vs. text-predicted shift)")
    ax.set_title("Component 2 diagnostic: calibration curve\n(green = trustworthy domains, red = untrustworthy)")
    ax.legend(fontsize=9)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "component2_calibration_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


def plot_variants_comparison():
    v = load("component2_defactify_grounding_variants.json")
    baseline = v["baseline"]["best_target_acc"]

    labels, gains, colors = [], [], []
    labels.append("+DDO\ntext-only\n(204 dir.)")
    gains.append(v["ddo_text"]["best_target_acc"] - baseline)
    colors.append("#3498db")

    for k in ["10", "20", "40", "60"]:
        labels.append(f"+DDO grounded\nmean, probe={k}")
        gains.append(v["ddo_grounded_mean_by_probe_size"][k]["best_target_acc"] - baseline)
        colors.append("#e74c3c")

    labels.append("+DDO grounded\npersample\n(probe=20)")
    gains.append(v["ddo_grounded_persample"]["best_target_acc"] - baseline)
    colors.append("#2ecc71")

    labels.append("+DDO grounded\nblend\n(probe=20)")
    gains.append(v["ddo_grounded_blend"]["best_target_acc"] - baseline)
    colors.append("#27ae60")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, gains, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)

    text_ddo_gain = v["ddo_text"]["best_target_acc"] - baseline
    noise_band = 0.3  # this project's own observed run-to-run spread on ddo_text (see report text)
    ax.axhspan(text_ddo_gain - noise_band, text_ddo_gain + noise_band, color="#3498db", alpha=0.12,
               label=f"noise band around text-DDO's gain (±{noise_band} pts, this project's\n"
                     f"own observed run-to-run spread on this exact measurement)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Best target (Midjourney v6) accuracy gain\nover baseline (percentage points)")
    ax.set_title("Component 2b: fallback design comparison\n(red = single mean direction, green = diversity-preserving fixes)")
    ax.legend(fontsize=8, loc="lower right")
    for xi, g in zip(x, gains):
        ax.text(xi, g + (0.05 if g >= 0 else -0.12), f"{g:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "component2b_variants_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


def plot_first_run_comparison():
    """The original 3-condition result (component2_self_diagnosing_grounding.md):
    baseline vs. text-only DDO vs. the single-mean-direction fallback that
    turned out to underperform both."""
    r = load("component2_defactify_grounding_ddo.json")
    conds = r["results_by_condition"]
    labels = ["Baseline\n(α=0)", "+DDO\ntext-only", "+DDO\ngrounded\n(single mean dir.)"]
    accs = [conds["baseline"]["best_target_acc"], conds["ddo_text"]["best_target_acc"],
            conds["ddo_grounded"]["best_target_acc"]]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    ax.bar(x, accs, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best target (Midjourney v6) accuracy (%)")
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    ax.set_title("Component 2, first run:\nthe single-mean-direction fallback underperformed both")
    for xi, a in zip(x, accs):
        ax.text(xi, a + 0.1, f"{a:.2f}%", ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "component2_first_run_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


def plot_seed_sweep_summary():
    """Component 2c: gain-over-baseline for the 4 key conditions across 3
    seeds on Midjourney v6 (same fixed data split, only training seed
    varies) plus 1 independent check on DALL-E 3 - shows both the average
    and, just as importantly, the spread of each fallback design."""
    mj = {
        "seed0": load("component2_defactify_grounding_variants.json"),
        "seed1": load("component2_defactify_grounding_variants_seed1.json"),
        "seed2": load("component2_defactify_grounding_variants_seed2.json"),
    }
    dalle3 = load("component2d_dalle3_validation.json")

    def gain(run, key):
        return run[key]["best_target_acc"] - run["baseline"]["best_target_acc"]

    conditions = [
        ("+DDO text-only", "ddo_text", None, "#3498db"),
        ("+DDO grounded\nmean (probe=20)", "ddo_grounded_mean_by_probe_size", "20", "#e74c3c"),
        ("+DDO grounded\npersample", "ddo_grounded_persample", None, "#2ecc71"),
        ("+DDO grounded\nblend", "ddo_grounded_blend", None, "#27ae60"),
    ]
    dalle3_key = {"ddo_text": "ddo_text", "ddo_grounded_mean_by_probe_size": "ddo_grounded_mean",
                  "ddo_grounded_persample": "ddo_grounded_persample", "ddo_grounded_blend": "ddo_grounded_blend"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_positions = np.arange(len(conditions))
    for xi, (label, key, subkey, color) in zip(x_positions, conditions):
        mj_gains = []
        for run in mj.values():
            entry = run[key][subkey] if subkey else run[key]
            mj_gains.append(entry["best_target_acc"] - run["baseline"]["best_target_acc"])
        dkey = dalle3_key[key]
        d_gain = dalle3["results_by_condition"][dkey]["best_target_acc"] - dalle3["results_by_condition"]["baseline"]["best_target_acc"]

        ax.scatter([xi] * len(mj_gains), mj_gains, color=color, s=70, zorder=3,
                   label="Midjourney v6, 3 seeds" if xi == 0 else None, marker="o")
        ax.scatter([xi], [d_gain], color=color, s=100, zorder=3, marker="^", edgecolor="black",
                   label="DALL-E 3, 1 seed" if xi == 0 else None)
        mean_all = np.mean(mj_gains + [d_gain])
        ax.hlines(mean_all, xi - 0.25, xi + 0.25, color=color, linewidth=2.5, zorder=2)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=9)
    ax.set_ylabel("Accuracy gain over baseline (percentage points)")
    ax.set_title("Component 2c: gain across 3 training seeds (Midjourney v6)\n+ 1 independent check (DALL-E 3) — thick line = mean")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "component2c_seed_sweep_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    plot_calibration_curve()
    plot_variants_comparison()
    plot_first_run_comparison()
    plot_seed_sweep_summary()
