"""Component 2 - threshold calibration.

Runs this project's own alignment-score diagnostic (model/domain_grounding.
compute_alignment_score) on PACS's photo -> {art_painting, cartoon, sketch}
domain shifts - domains already established (Component 1's own core
result) to sit inside CLIP's comfort zone - as the trust-positive
calibration point for Component 2's fallback threshold. This is the
apples-to-apples counterpart to Phase F3's already-measured 0.037 mean
alignment for photo -> Midjourney v6 (an untrustworthy shift): same
formula (real matched photos on both sides), same project, just a
different, known-good domain shift, so the threshold is grounded in a real
separation rather than borrowed from the paper's differently-computed
0.90-0.99 claim.

Uses a small probe sample per class/domain (not PACS's full ~2,000-image
domains) - the whole point of Component 2's diagnostic is that it must be
cheap enough to run the moment a new domain arrives, before any full
training run.
"""
import os
import sys
import json
import random
from collections import defaultdict

import torch
import clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from data.PACS.pacs_data import Processed_PACS_Dataset
from model.domain_grounding import compute_alignment_score

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
PROBE_PER_CLASS = 30
SEED = 0

SOURCE_DOMAIN = "photo"
SOURCE_TEMPLATE = "a photo of a {}."
TARGET_DOMAINS = {
    "art_painting": "a painting of a {}.",
    "cartoon": "a cartoon of a {}.",
    "sketch": "a sketch of a {}.",
}

# Phase F3's already-measured number for the untrustworthy anchor
# (photo -> Midjourney v6, real matched photos, same formula) - cited here
# for the write-up's threshold justification, not recomputed by this script.
PHASE_F3_MEAN_ALIGNMENT = 0.037


def sample_indices_by_class(dataset, samples_per_class, seed=SEED):
    rng = random.Random(seed)
    idx_by_class = defaultdict(list)
    for i, line in enumerate(dataset.annos):
        _path, cls_label = line.strip().split(",")
        idx_by_class[int(cls_label)].append(i)
    sampled = {}
    for label, idxs in idx_by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        sampled[label] = idxs[:samples_per_class]
    return sampled


def build_images_by_class(dataset, class_names, id_by_name, samples_per_class):
    sampled = sample_indices_by_class(dataset, samples_per_class)
    images_by_class = {}
    for c in class_names:
        label = id_by_name[c]
        images_by_class[c] = [dataset[i][0] for i in sampled[label]]  # already-preprocessed tensors
    return images_by_class


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, _preprocess = clip.load(args.CLIP_type, device=args.device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    data_root = os.path.join(args.data_dir, "PACS", "pacs_raw", "kfold")

    print(f"Loading PACS source domain '{SOURCE_DOMAIN}' (train split)...")
    source_ds = Processed_PACS_Dataset(
        args, data_root=data_root, domain=SOURCE_DOMAIN, split="train", meta_root="data/PACS",
    )
    class_names = list(source_ds.classname2id.keys())
    source_images_by_class = build_images_by_class(source_ds, class_names, source_ds.classname2id, PROBE_PER_CLASS)
    print(f"Classes: {class_names}")
    print(f"Probe size: {PROBE_PER_CLASS}/class x {len(class_names)} classes = "
          f"{PROBE_PER_CLASS * len(class_names)} source images")

    results_per_domain = {}
    for target_domain, target_template in TARGET_DOMAINS.items():
        print(f"\nLoading PACS target domain '{target_domain}' (train split)...")
        target_ds = Processed_PACS_Dataset(
            args, data_root=data_root, domain=target_domain, split="train", meta_root="data/PACS",
            classname2id=source_ds.classname2id, concept2id=source_ds.concept2id,
        )
        target_images_by_class = build_images_by_class(
            target_ds, class_names, target_ds.classname2id, PROBE_PER_CLASS
        )

        per_class_alignment, mean_alignment = compute_alignment_score(
            clip_model, source_images_by_class, target_images_by_class,
            class_names, SOURCE_TEMPLATE, target_template, args.device,
        )
        print(f"  photo -> {target_domain}: mean alignment = {mean_alignment:.4f}")
        print(f"  per-class: {per_class_alignment}")
        results_per_domain[target_domain] = {
            "target_template": target_template,
            "per_class_alignment": per_class_alignment,
            "mean_alignment": mean_alignment,
        }

    overall_mean = sum(r["mean_alignment"] for r in results_per_domain.values()) / len(results_per_domain)
    # Threshold: midpoint between PACS's own measured mean (trust-positive
    # anchor, this exact diagnostic code) and Phase F3's already-measured
    # photo->Midjourney number (trust-negative anchor, same formula).
    threshold = (overall_mean + PHASE_F3_MEAN_ALIGNMENT) / 2

    print(f"\nOverall PACS mean alignment (3 domains): {overall_mean:.4f}")
    print(f"Phase F3 photo->Midjourney v6 mean alignment (cited, same formula): {PHASE_F3_MEAN_ALIGNMENT}")
    print(f"Calibrated threshold (midpoint): {threshold:.4f}")

    results = {
        "clip_type": args.CLIP_type,
        "probe_per_class": PROBE_PER_CLASS,
        "source_domain": SOURCE_DOMAIN,
        "source_template": SOURCE_TEMPLATE,
        "class_names": class_names,
        "results_per_target_domain": results_per_domain,
        "pacs_overall_mean_alignment": overall_mean,
        "phase_f3_mean_alignment_cited": PHASE_F3_MEAN_ALIGNMENT,
        "calibrated_threshold": threshold,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component2_alignment_calibration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
