"""Simple bar chart for Phase E1's descriptor-pool term check.
Usage: python generate_phase_e1_figure.py
"""
import os
import json
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "phase_e1_results.json")) as f:
    r = json.load(f)

labels = [
    "Direct AI-generation terms\n(\"Stable Diffusion\", \"Midjourney\", \"AI-generated\"...)",
    "Phase F3's 5 specific generators\nnamed anywhere in the pool",
    "Conceptually-adjacent terms\n(\"digital art\", \"CGI render\", \"cyberpunk\"...)",
]
found = [r["n_direct_ai_term_matches"], r["n_generators_named"], r["n_adjacent_term_matches"]]
checked = [20, r["n_generators_checked"], 10]
colors = ["#c0392b", "#c0392b", "#2471a3"]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
y = range(len(labels))
bars = ax.barh(y, found, color=colors, height=0.5)
for i, (f_val, c_val) in enumerate(zip(found, checked)):
    ax.text(f_val + 0.3, i, f"{f_val} / {c_val}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count found in LanCE's actual 204-entry descriptor pool")
ax.set_xlim(0, 21)
ax.invert_yaxis()
ax.set_title("Phase E1: what's actually in LanCE's frozen, GPT-3.5-written descriptor list")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "phase_e1_term_check.png"), dpi=150)
plt.close(fig)
print("Wrote phase_e1_term_check.png")
