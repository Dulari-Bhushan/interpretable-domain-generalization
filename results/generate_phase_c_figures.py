"""Plot Phase C's remediation comparison from results/phase_b_results.json
(naive-sequential baseline) and results/phase_c_results.json (cumulative-DDO,
replay, EWC), all sharing the same 3 domain orderings.

Three figures:
  1. phase_c_bwt_comparison.png - BWT per ordering, grouped by condition
     (baseline + 3 remediations) - the headline "did it fix it" figure.
  2. phase_c_acc_comparison.png - ACC_final per ordering, grouped by
     condition - the accuracy-cost/benefit tradeoff.
  3. phase_c_photo_first_forgetting.png - accuracy on 'photo' across stages
     for the photo_art_cartoon_sketch ordering specifically (the only
     ordering with real baseline forgetting), one line per condition.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(REPO_ROOT, "results", "phase_b_results.json")) as f:
    baseline = json.load(f)
with open(os.path.join(REPO_ROOT, "results", "phase_c_results.json")) as f:
    remediations = json.load(f)

CONDITIONS = ["baseline", "cumulative_ddo", "replay", "ewc"]
COLORS = {"baseline": "#7f8c8d", "cumulative_ddo": "#2471a3", "replay": "#27ae60", "ewc": "#c0392b"}

runs_by_condition = {"baseline": {r["order_id"]: r for r in baseline["sequential_runs"]}}
for name, runs in remediations["remediations"].items():
    runs_by_condition[name] = {r["order_id"]: r for r in runs}

order_ids = [r["order_id"] for r in baseline["sequential_runs"]]

# --- 1. BWT comparison ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(order_ids))
width = 0.2
for i, cond in enumerate(CONDITIONS):
    vals = [runs_by_condition[cond][oid]["BWT"] for oid in order_ids]
    ax.bar(x + (i - 1.5) * width, vals, width, label=cond, color=COLORS[cond])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(order_ids, rotation=15, ha="right")
ax.set_ylabel("BWT (percentage points)")
ax.set_title("Phase C: does a textbook CL fix close the forgetting gap?\n(negative = forgetting)")
ax.legend()
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_c_bwt_comparison.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

# --- 2. ACC comparison ---
fig, ax = plt.subplots(figsize=(9, 5))
for i, cond in enumerate(CONDITIONS):
    vals = [runs_by_condition[cond][oid]["ACC_final"] for oid in order_ids]
    ax.bar(x + (i - 1.5) * width, vals, width, label=cond, color=COLORS[cond])
ax.axhline(baseline["joint_oracle"]["ACC"], color="black", linestyle="--", linewidth=1,
           label=f"joint/oracle ({baseline['joint_oracle']['ACC']:.1f}%)")
ax.set_xticks(x)
ax.set_xticklabels(order_ids, rotation=15, ha="right")
ax.set_ylabel("Final ACC (%)")
ax.set_ylim(75, 102)
ax.set_title("Phase C: accuracy cost/benefit of each remediation")
ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_c_acc_comparison.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

# --- 3. Photo-first forgetting curve, all conditions ---
fig, ax = plt.subplots(figsize=(7, 4.5))
target_order = "photo_art_cartoon_sketch"
for cond in CONDITIONS:
    run = runs_by_condition[cond][target_order]
    stages = run["stages"]
    xs = [s["stage"] for s in stages]
    ys = [s["per_domain_test_acc"]["photo"] for s in stages]
    ax.plot(xs, ys, "o-", label=cond, color=COLORS[cond])
ax.axhline(baseline["joint_oracle"]["per_domain_test_acc"]["photo"], linestyle="--", color="gray", alpha=0.6,
           label="joint/oracle")
ax.set_xlabel("Training stage")
ax.set_ylabel("Accuracy on 'photo' (%)")
ax.set_title(f"Phase C: the one ordering with real baseline forgetting\n({target_order}, domain 1 = photo)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_c_photo_first_forgetting.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
