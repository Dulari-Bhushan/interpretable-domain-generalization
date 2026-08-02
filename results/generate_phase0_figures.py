"""Parse Phase 0 training logs and plot baseline-vs-DDO accuracy curves.

Usage: python generate_phase0_figures.py
Reads external/LanCE/logs_baseline_run.log and logs_ddo_run.log (repo-relative),
writes PNGs to results/figures/.
"""
import re
import os
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANCE_DIR = os.path.join(REPO_ROOT, "external", "LanCE")
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EPOCH_RE = re.compile(r"Epoch (\d+) Summary:")
TRAIN_RE = re.compile(r"Train Acc:\s*([\d.]+)%")
SOURCE_RE = re.compile(r"Source Acc:\s*([\d.]+)%")
TARGET_RE = re.compile(r"Target Acc:\s*([\d.]+)%")


def parse_log(path):
    """Handles both main.py's multi-line-per-epoch format and
    train_cached.py's single-line-per-epoch format."""
    epochs, train_acc, source_acc, target_acc = [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Single-line format (train_cached.py): one line has epoch+train+source+target all together
    single_line = re.compile(
        r"Epoch (\d+) Summary: Time: [\d.]+s \| Train Acc: ([\d.]+)% \| "
        r"Source Acc: ([\d.]+)% \| Target Acc: ([\d.]+)%"
    )
    single_matches = single_line.findall(text)
    if single_matches:
        for ep, tr, so, ta in single_matches:
            epochs.append(int(ep))
            train_acc.append(float(tr))
            source_acc.append(float(so))
            target_acc.append(float(ta))
        return epochs, train_acc, source_acc, target_acc

    # Multi-line format (main.py): Epoch/Train/Source/Target on separate consecutive lines
    lines = text.splitlines()
    cur_epoch = None
    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            cur_epoch = int(m.group(1))
            continue
        m = TRAIN_RE.search(line)
        if m and cur_epoch is not None:
            train_acc.append(float(m.group(1)))
            epochs.append(cur_epoch)
            continue
        m = SOURCE_RE.search(line)
        if m and cur_epoch is not None:
            source_acc.append(float(m.group(1)))
            continue
        m = TARGET_RE.search(line)
        if m and cur_epoch is not None:
            target_acc.append(float(m.group(1)))
            cur_epoch = None  # reset; next Epoch line starts a new block
    return epochs, train_acc, source_acc, target_acc


def main():
    baseline_path = os.path.join(LANCE_DIR, "logs_baseline_run.log")
    ddo_path = os.path.join(LANCE_DIR, "logs_ddo_run.log")

    b_ep, b_tr, b_so, b_ta = parse_log(baseline_path)
    d_ep, d_tr, d_so, d_ta = parse_log(ddo_path)

    # --- Figure 1: target (OOD) accuracy, the headline comparison ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(b_ep, b_ta, label="baseline (α=0)", color="#c0392b", linewidth=2)
    ax.plot(d_ep, d_ta, label="+DDO (α=1)", color="#2471a3", linewidth=2)
    ax.axhline(50.54, color="#c0392b", linestyle="--", alpha=0.5, linewidth=1, label="paper baseline (50.54%)")
    ax.axhline(55.53, color="#2471a3", linestyle="--", alpha=0.5, linewidth=1, label="paper +DDO (55.53%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Target (CUB-Painting) accuracy (%)")
    ax.set_title("Phase 0: LanCE reproduction on CUB → CUB-Painting")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase0_target_accuracy.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: all three splits, both conditions, small multiples ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    panels = [("Train", b_tr, d_tr), ("Source (ID) test", b_so, d_so), ("Target (OOD) test", b_ta, d_ta)]
    for ax, (title, b_series, d_series) in zip(axes, panels):
        ax.plot(b_ep, b_series, label="baseline (α=0)", color="#c0392b", linewidth=2)
        ax.plot(d_ep, d_series, label="+DDO (α=1)", color="#2471a3", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Phase 0: full accuracy curves, CUB / CUB-Painting")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase0_all_accuracy_curves.png"), dpi=150)
    plt.close(fig)

    print(f"Baseline: {len(b_ep)} epochs parsed, final target acc {b_ta[-1] if b_ta else 'N/A'}%")
    print(f"DDO: {len(d_ep)} epochs parsed, final target acc {d_ta[-1] if d_ta else 'N/A'}%")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
