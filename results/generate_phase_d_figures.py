"""Plot Phase D's Office-Home results from results/phase_d_results.json
(naive-sequential baseline) and results/phase_d_remediation_results.json
(cumulative-DDO, replay, EWC), plus a direct comparison against Phase B's
PACS baseline (results/phase_b_results.json) - the whole point of Phase D
was to check whether PACS's near-ceiling accuracy was masking forgetting
that a harder, more class-crowded benchmark would reveal.

Four figures:
  1. phase_d_acc_heatmap_<order_id>.png - stage x evaluated-domain accuracy
     matrix per ordering (baseline only), same convention as Phase B.
  2. phase_d_bwt_comparison.png - BWT per ordering, grouped by condition
     (baseline + 3 remediations).
  3. phase_d_acc_comparison.png - ACC_final per ordering, grouped by condition.
  4. phase_d_vs_phase_b_bwt.png - baseline BWT side-by-side, PACS vs
     Office-Home, matched by ordering rank (order 1/2/3) - the headline
     "did the harder benchmark reveal more forgetting" figure.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(REPO_ROOT, "results", "phase_d_results.json")) as f:
    baseline = json.load(f)
with open(os.path.join(REPO_ROOT, "results", "phase_d_remediation_results.json")) as f:
    remediations = json.load(f)
with open(os.path.join(REPO_ROOT, "results", "phase_b_results.json")) as f:
    pacs_baseline = json.load(f)

# --- 1. Per-ordering accuracy heatmaps (baseline) ---
for run in baseline["sequential_runs"]:
    order = run["domain_order"]
    stages = run["stages"]
    matrix = np.array([[s["per_domain_test_acc"][d] for d in order] for s in stages])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([f"stage {s['stage']} ({s['trained_domain']})" for s in stages])
    ax.set_xlabel("Evaluated domain")
    ax.set_ylabel("Training stage")
    ax.set_title(f"Phase D (Office-Home) accuracy matrix: {run['order_id']}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                     color="white" if matrix[i, j] < 60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Test accuracy (%)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"phase_d_acc_heatmap_{run['order_id']}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)

# --- 2/3. BWT and ACC comparison across conditions ---
CONDITIONS = ["baseline", "cumulative_ddo", "replay", "ewc"]
COLORS = {"baseline": "#7f8c8d", "cumulative_ddo": "#2471a3", "replay": "#27ae60", "ewc": "#c0392b"}

runs_by_condition = {"baseline": {r["order_id"]: r for r in baseline["sequential_runs"]}}
for name, runs in remediations["remediations"].items():
    runs_by_condition[name] = {r["order_id"]: r for r in runs}
order_ids = [r["order_id"] for r in baseline["sequential_runs"]]

x = np.arange(len(order_ids))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))
for i, cond in enumerate(CONDITIONS):
    vals = [runs_by_condition[cond][oid]["BWT"] for oid in order_ids]
    ax.bar(x + (i - 1.5) * width, vals, width, label=cond, color=COLORS[cond])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(order_ids, rotation=15, ha="right")
ax.set_ylabel("BWT (percentage points)")
ax.set_title("Phase D (Office-Home): does a textbook CL fix close the forgetting gap?\n(negative = forgetting)")
ax.legend()
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_d_bwt_comparison.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

fig, ax = plt.subplots(figsize=(10, 5))
for i, cond in enumerate(CONDITIONS):
    vals = [runs_by_condition[cond][oid]["ACC_final"] for oid in order_ids]
    ax.bar(x + (i - 1.5) * width, vals, width, label=cond, color=COLORS[cond])
ax.axhline(baseline["joint_oracle"]["ACC"], color="black", linestyle="--", linewidth=1,
           label=f"joint/oracle ({baseline['joint_oracle']['ACC']:.1f}%)")
ax.set_xticks(x)
ax.set_xticklabels(order_ids, rotation=15, ha="right")
ax.set_ylabel("Final ACC (%)")
ax.set_title("Phase D (Office-Home): accuracy cost/benefit of each remediation")
ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_d_acc_comparison.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

# --- 4. PACS vs Office-Home baseline BWT, matched by ordering rank ---
pacs_bwts = [r["BWT"] for r in pacs_baseline["sequential_runs"]]
oh_bwts = [r["BWT"] for r in baseline["sequential_runs"]]
labels = [f"ordering {i+1}" for i in range(len(pacs_bwts))]

fig, ax = plt.subplots(figsize=(7, 4.5))
xx = np.arange(len(labels))
ax.bar(xx - 0.2, pacs_bwts, 0.4, label="PACS (Phase B)", color="#9b59b6")
ax.bar(xx + 0.2, oh_bwts, 0.4, label="Office-Home (Phase D)", color="#e67e22")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylabel("BWT (percentage points)")
ax.set_title("Did a harder benchmark reveal more forgetting?\nnaive-sequential baseline, PACS vs Office-Home")
ax.legend()
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_d_vs_phase_b_bwt.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
