"""Phase F1 - Domain-shift alignment check on EuroSAT (Pillar 2, secondary).

Pillar 2's question is different from Pillar 1's (Phases 0-D): even a
perfectly-solved forgetting problem wouldn't save LanCE in the long run,
because its entire mechanism depends on CLIP's frozen image-text alignment
(Sec. 3) - and that alignment is demonstrated (their Fig. 2/8) only for
domains heavily represented in CLIP's training data (sketch, painting,
sculpture, clipart, ...). Satellite imagery is a domain CLIP is known to
align poorly with (OpenAI's own CLIP paper: ViT-L/14-336 zero-shot on
EuroSAT = 59.6%, vs. a linear probe on the identical visual features =
98.1% - a 38.5-point gap showing the *visual* representation is fine, the
failure is specifically in *image-text alignment*, the one mechanism DDO
depends on entirely).

What this script computes (the piece that isn't already published and
needs to be measured ourselves): a domain-shift alignment score in the
same style as the paper's own Fig. 2 methodology - cosine similarity
between a *visual* domain-shift direction and a *textual* domain-shift
direction, for a photo -> satellite shift on EuroSAT's 10 land-cover
classes, to compare directly against their own reported 0.90-0.99
alignment range for sketch/sculpture/painting shifts.

Adaptation note (documented per this project's convention of stating
deviations honestly): the paper's Fig. 2 visual shift is real-photo-image
embedding vs. real-target-domain-image embedding. We don't have a matched
"ground photo of a forest/highway/..." image dataset, and sourcing one
would turn this from "cheap, ~1-2 hours, minimal setup" into a second
dataset-acquisition project. Instead we use CLIP's own text embedding of
"a photo of a {class}." as the photo-domain reference point in CLIP's
shared embedding space, justified by CLIP's own extremely well-documented
near-unity image-text alignment for exactly this kind of everyday-photo
domain (their own Fig. 2 baseline, plus the general CLIP zero-shot
literature - this is also literally the same assumption this codebase's
own domain_diffs machinery relies on everywhere else, e.g.
source_text_prompts = ['a photo of a {}.'], never a real photo image).
So: visual_shift(cls) = mean_image_embedding(EuroSAT satellite images of
cls) - text_embedding("a photo of a {cls}."); textual_shift(cls) =
text_embedding("a satellite image of a {cls}.") - text_embedding("a photo
of a {cls}."); alignment(cls) = cosine_similarity(visual_shift, textual_shift).

Also computes our own zero-shot classification accuracy on EuroSAT (same
CLIP ViT-L/14 used throughout this project, not the ViT-L/14-336px OpenAI
reported 59.6% for) as a second, independently interpretable signal and a
sanity check against the published number.

No training run - matches the plan's "cheap, minimal setup" framing.
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

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
EUROSAT_ROOT = "data/EuroSAT_raw"
SAMPLES_PER_CLASS = 200  # "we only need a small sample" per the plan
SEED = 0

# LanCE paper's own reported alignment range for domains it *does* claim to
# handle well (Fig. 2, sketch/sculpture/painting shifts) - our comparison anchor.
PAPER_ALIGNMENT_RANGE = (0.90, 0.99)
# OpenAI CLIP paper's own published numbers, ViT-L/14-336px, EuroSAT (not
# reproduced here - cited directly, "already-published, unimpeachable").
PAPER_ZEROSHOT_ACC_336PX = 59.6
PAPER_LINEAR_PROBE_ACC_336PX = 98.1

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
    class_names = ds.classes  # folder-order, index-aligned with ds.class_to_idx
    display_names = [CLASS_DISPLAY_NAMES[c] for c in class_names]

    sampled = sample_indices_per_class(ds, SAMPLES_PER_CLASS)
    print(f"Sampled {SAMPLES_PER_CLASS}/class across {len(class_names)} classes "
          f"({sum(len(v) for v in sampled.values())} images total)")

    # Text embeddings: "a photo of a {}." (photo-domain proxy) and
    # "a satellite image of a {}." (true target domain), per class.
    with torch.no_grad():
        photo_texts = clip.tokenize([f"a photo of a {c}." for c in display_names]).to(args.device)
        photo_text_emb = clip_model.encode_text(photo_texts).float()
        photo_text_emb = photo_text_emb / photo_text_emb.norm(dim=-1, keepdim=True)

        satellite_texts = clip.tokenize([f"a satellite image of a {c}." for c in display_names]).to(args.device)
        satellite_text_emb = clip_model.encode_text(satellite_texts).float()
        satellite_text_emb = satellite_text_emb / satellite_text_emb.norm(dim=-1, keepdim=True)

    # Mean image embedding per class (from sampled EuroSAT satellite images),
    # and zero-shot classification accuracy against the satellite-domain
    # text prompts (computed on the same sample).
    per_class_alignment = {}
    correct, total = 0, 0
    with torch.no_grad():
        for label in range(len(class_names)):
            idxs = sampled[label]
            imgs = torch.stack([ds[i][0] for i in idxs]).to(args.device)
            img_emb = clip_model.encode_image(imgs).float()
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            mean_img_emb = img_emb.mean(dim=0)
            mean_img_emb = mean_img_emb / mean_img_emb.norm()

            visual_shift = mean_img_emb - photo_text_emb[label]
            visual_shift = visual_shift / visual_shift.norm()
            textual_shift = satellite_text_emb[label] - photo_text_emb[label]
            textual_shift = textual_shift / textual_shift.norm()

            alignment = torch.dot(visual_shift, textual_shift).item()
            per_class_alignment[class_names[label]] = alignment

            sims = img_emb @ satellite_text_emb.T  # (n_samples, n_classes)
            preds = sims.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += len(idxs)

    mean_alignment = sum(per_class_alignment.values()) / len(per_class_alignment)
    our_zeroshot_acc = 100 * correct / total

    print(f"\nPer-class alignment: {per_class_alignment}")
    print(f"Mean alignment (photo -> satellite shift): {mean_alignment:.4f}")
    print(f"Paper's own reported range for sketch/sculpture/painting shifts: {PAPER_ALIGNMENT_RANGE}")
    print(f"\nOur zero-shot EuroSAT accuracy ({args.CLIP_type}): {our_zeroshot_acc:.2f}%")
    print(f"OpenAI's published zero-shot (ViT-L/14-336px): {PAPER_ZEROSHOT_ACC_336PX}%")
    print(f"OpenAI's published linear-probe ceiling (ViT-L/14-336px): {PAPER_LINEAR_PROBE_ACC_336PX}%")

    results = {
        "clip_type": args.CLIP_type,
        "samples_per_class": SAMPLES_PER_CLASS,
        "classes": class_names,
        "display_names": display_names,
        "per_class_alignment": per_class_alignment,
        "mean_alignment": mean_alignment,
        "paper_alignment_range": PAPER_ALIGNMENT_RANGE,
        "our_zeroshot_acc": our_zeroshot_acc,
        "paper_zeroshot_acc_336px": PAPER_ZEROSHOT_ACC_336PX,
        "paper_linear_probe_acc_336px": PAPER_LINEAR_PROBE_ACC_336PX,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_f1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
