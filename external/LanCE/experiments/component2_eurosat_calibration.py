"""Component 2 follow-up #4 (results/component2_self_diagnosing_grounding.md
"What's next"): remeasure EuroSAT's photo -> satellite alignment with this
project's own probe-scale diagnostic (model/domain_grounding.py), the same
code used for the PACS and Defactify calibration points, for a 3-point
calibration curve (PACS 0.249 real-photo / Defactify 0.037 real-photo /
EuroSAT here) instead of citing Phase F1's separately-implemented number.

EuroSAT has no matched real "photo of a forest/highway/..." dataset (same
gap Phase F1 hit), so this uses the same text-proxy adaptation Phase F1
used - domain_grounding.compute_alignment_score's source_images_by_class=
None mode, added specifically to let this script reuse the diagnostic
rather than reimplementing Phase F1's formula a second time. No training
run: EuroSAT's classes aren't in the same label space as Defactify/PACS
(this is a calibration/diagnostic check only, per the plan's own scope
note - see planning/02-continual-dg-experiment-plan.md's Phase F2
discussion of why EuroSAT isn't a domain-generalization test).
"""
import os
import sys
import json
import random
from collections import defaultdict

import torch
import clip
from torchvision.datasets import EuroSAT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from model.domain_grounding import compute_alignment_score

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
EUROSAT_ROOT = "data/EuroSAT_raw"
PROBE_PER_CLASS = 30  # matches component2_alignment_calibration.py's PACS probe size
SEED = 0
SOURCE_TEMPLATE = "a photo of a {}."
TARGET_TEMPLATE = "a satellite image of a {}."

# Phase F1's own already-measured number (200 images/class, same text-proxy
# formula) - cited for a direct probe-scale-vs-full-sample consistency check.
PHASE_F1_MEAN_ALIGNMENT = 0.324

CLASS_DISPLAY_NAMES = {
    "AnnualCrop": "annual crop field",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway",
    "Industrial": "industrial area",
    "Pasture": "pasture",
    "PermanentCrop": "permanent crop field",
    "Residential": "residential area",
    "River": "river",
    "SeaLake": "sea or lake",
}


def sample_indices_per_class(dataset, samples_per_class, seed=SEED):
    rng = random.Random(seed)
    idx_by_class = defaultdict(list)
    for i, (_path, label) in enumerate(dataset.samples):
        idx_by_class[label].append(i)
    sampled = {}
    for label, idxs in idx_by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        sampled[label] = idxs[:samples_per_class]
    return sampled


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(args.CLIP_type, device=args.device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    ds = EuroSAT(root=EUROSAT_ROOT, download=True, transform=preprocess)
    class_names_raw = ds.classes
    display_names = [CLASS_DISPLAY_NAMES[c] for c in class_names_raw]

    sampled = sample_indices_per_class(ds, PROBE_PER_CLASS)
    print(f"Probe: {PROBE_PER_CLASS}/class x {len(class_names_raw)} classes "
          f"= {sum(len(v) for v in sampled.values())} images")

    target_images_by_class = {}
    for label, name in enumerate(display_names):
        idxs = sampled[label]
        target_images_by_class[name] = [ds[i][0] for i in idxs]  # already preprocessed (EuroSAT applies transform)

    per_class_alignment, mean_alignment = compute_alignment_score(
        clip_model, None, target_images_by_class, display_names,
        SOURCE_TEMPLATE, TARGET_TEMPLATE, args.device,
    )

    print(f"\nPer-class alignment: {per_class_alignment}")
    print(f"Mean alignment (probe={PROBE_PER_CLASS}/class): {mean_alignment:.4f}")
    print(f"Phase F1's own number (200/class, same formula): {PHASE_F1_MEAN_ALIGNMENT}")

    results = {
        "clip_type": args.CLIP_type,
        "probe_per_class": PROBE_PER_CLASS,
        "display_names": display_names,
        "per_class_alignment": per_class_alignment,
        "mean_alignment": mean_alignment,
        "phase_f1_mean_alignment_200_per_class": PHASE_F1_MEAN_ALIGNMENT,
        "source_mode": "text_proxy (no matched real photo dataset, same adaptation as Phase F1)",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component2_eurosat_calibration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
