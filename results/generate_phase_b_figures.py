"""Plot Phase B's Domain-IL sequential-forgetting results from
results/phase_b_results.json (or --input, e.g. the smoke-test JSON).

Four figures:
  1. phase_b_acc_heatmap_<order_id>.png - stage x evaluated-domain accuracy
     matrix per ordering (the classic CL accuracy matrix).
  2. phase_b_bwt_comparison.png - BWT per ordering, joint/oracle ACC as reference.
  3. phase_b_forgetting_curves.png - accuracy on domain_order[0] across stages,
     one line per ordering, joint/oracle accuracy on that domain as reference.
  4. phase_b_ddo_erosion.png - DDO mechanism metric across stages, one line
     per ordering.
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=os.path.join(REPO_ROOT, "results", "phase_b_results.json"))
args = parser.parse_args()

os.makedirs(FIG_DIR, exist_ok=True)

with open(args.input) as f:
    data = json.load(f)

domains = data["domains"]
joint_acc = data["joint_oracle"]["per_domain_test_acc"]
joint_ACC = data["joint_oracle"]["ACC"]
runs = data["sequential_runs"]

# --- 1. Per-ordering accuracy heatmaps ---
for run in runs:
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
    ax.set_title(f"Phase B accuracy matrix: {run['order_id']}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                     color="white" if matrix[i, j] < 60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Test accuracy (%)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"phase_b_acc_heatmap_{run['order_id']}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)

# --- 2. BWT comparison ---
fig, ax = plt.subplots(figsize=(7, 4.5))
order_ids = [r["order_id"] for r in runs]
bwts = [r["BWT"] for r in runs]
colors = ["#c0392b" if b < 0 else "#2471a3" for b in bwts]
ax.bar(order_ids, bwts, color=colors)
ax.axhline(0, color="#7f8c8d", linewidth=1)
ax.set_ylabel("BWT (percentage points)")
ax.set_title("Phase B: Backward Transfer by domain ordering\n(negative = forgetting)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_b_bwt_comparison.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

# --- 3. Domain-1 forgetting curves ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for run in runs:
    order = run["domain_order"]
    domain_1 = order[0]
    stages = run["stages"]
    xs = [s["stage"] for s in stages]
    ys = [s["per_domain_test_acc"][domain_1] for s in stages]
    ax.plot(xs, ys, "o-", label=f"{run['order_id']} (domain 1 = {domain_1})")
    ax.axhline(joint_acc[domain_1], linestyle="--", alpha=0.3, color="gray")
ax.set_xlabel("Training stage")
ax.set_ylabel("Accuracy on domain 1 (%)")
ax.set_title("Phase B: does domain-1 accuracy survive later stages?\n(dashed = joint/oracle accuracy on that domain)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_b_forgetting_curves.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)

# --- 4. DDO erosion over stages ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for run in runs:
    stages = run["stages"]
    xs = [-1] + [s["stage"] for s in stages]  # -1 = pre-training baseline
    ys = [run["pre_training_ddo_erosion"]] + [s["ddo_erosion"] for s in stages]
    ax.plot(xs, ys, "o-", label=run["order_id"])
ax.set_xlabel("Training stage (-1 = pre-training, random init)")
ax.set_ylabel("DDO orthogonality loss on domain-1 descriptors\n(current weights)")
ax.set_title("Phase B: does the orthogonality property erode across stages?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, "phase_b_ddo_erosion.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
