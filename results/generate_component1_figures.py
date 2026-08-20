"""Plot Component 1's exact-match result across all three datasets/scales,
from results/phase_b_results.json, phase_d_results.json (original SGD
baselines), and component1_{pacs,officehome,domainnet}_results.json
(Component 1's analytic classifier).

One figure: BWT by domain ordering, grouped by dataset, original SGD vs.
Component 1 - the whole point being that Component 1's BWT stays near zero
regardless of dataset or scale, while the original SGD baseline's forgetting
gets worse as the benchmark gets harder (PACS -> Office-Home). DomainNet has
no original-SGD run (Component 1 is the only thing ever trained on it), so
it's shown with only the Component 1 bar, not a missing/zero SGD bar.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


phase_b = load("phase_b_results.json")
phase_d = load("phase_d_results.json")
c1_pacs = load("component1_pacs_results.json")
c1_oh = load("component1_officehome_results.json")
c1_dn = load("component1_domainnet_results.json")

sgd_bwt_by_order = {}
for run in phase_b["sequential_runs"]:
    sgd_bwt_by_order[("PACS", run["order_id"])] = run["BWT"]
for run in phase_d["sequential_runs"]:
    sgd_bwt_by_order[("OfficeHome", run["order_id"])] = run["BWT"]

rows = []  # (dataset, order_id, sgd_bwt_or_None, c1_bwt)
for run in c1_pacs["sequential_runs"]:
    rows.append(("PACS", run["order_id"], sgd_bwt_by_order.get(("PACS", run["order_id"])), run["BWT"]))
for run in c1_oh["sequential_runs"]:
    rows.append(("OfficeHome", run["order_id"], sgd_bwt_by_order.get(("OfficeHome", run["order_id"])), run["BWT"]))
for run in c1_dn["sequential_runs"]:
    rows.append(("DomainNet", run["order_id"], None, run["BWT"]))

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(rows))
width = 0.35

sgd_vals = [r[2] if r[2] is not None else np.nan for r in rows]
c1_vals = [r[3] for r in rows]

ax.bar(x - width / 2, sgd_vals, width, label="Original SGD (no SGD run exists for DomainNet)", color="#c0392b")
ax.bar(x + width / 2, c1_vals, width, label="Component 1 (analytic)", color="#2471a3")
ax.axhline(0, color="#7f8c8d", linewidth=1)

dataset_boundaries = []
prev_ds = None
for i, r in enumerate(rows):
    if r[0] != prev_ds:
        dataset_boundaries.append((i, r[0]))
        prev_ds = r[0]
for pos, _ in dataset_boundaries[1:]:
    ax.axvline(pos - 0.5, color="black", linewidth=0.8, linestyle=":")

ax.set_xticks(x)
ax.set_xticklabels([f"{r[0]}\n{r[1][:22]}" for r in rows], fontsize=7, rotation=45, ha="right")
ax.set_ylabel("BWT (percentage points, negative = forgetting)")
ax.set_title("Component 1: BWT stays near zero regardless of dataset scale\n"
              "(PACS: 4 domains/7 classes -> Office-Home: 4/65 -> DomainNet: 6/345)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
out = os.path.join(FIG_DIR, "component1_bwt_by_scale.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("Wrote", out)
