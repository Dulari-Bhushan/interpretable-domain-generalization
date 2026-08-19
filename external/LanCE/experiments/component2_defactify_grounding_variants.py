"""Component 2 follow-ups (results/component2_self_diagnosing_grounding.md
"What's next" #1-3), on the same Defactify (photo -> Midjourney v6) setup
as component2_defactify_grounding_ddo.py.

That first run found the diagnostic worked (correctly flagged the domain as
untrustworthy) but its single-mean-direction fallback made target accuracy
WORSE than baseline (73.81% vs. 74.27%), worse than text-only DDO's 75.03%.
This script tests three hypotheses for why, each a variant of the fallback:

1. ddo_grounded_persample: keep every probe image as its own direction
   instead of collapsing to one mean - tests whether the harm came from
   losing directional diversity (204 text directions -> 1 image direction).
2. ddo_grounded_mean_probe{10,40,60}: sweep probe size for the ORIGINAL
   single-mean-direction design - tests whether the harm is a small-sample
   noise artifact that shrinks with more probe images. Uses a FIXED
   target_test set (held out beyond the largest probe, 60/class) across all
   probe sizes so only the probe's own size varies, not what's evaluated on.
3. ddo_grounded_blend: text_domain_diffs (204 directions) concatenated with
   the mean-direction fallback (1 direction), rather than replacing text
   with images - tests whether the harm is about REMOVING the text pool's
   diversity, independent of whether the image direction itself is useful.

baseline and ddo_text are trained once (they don't depend on probe size)
against the fixed target_test set and reused for every comparison below.
"""
import os
import sys
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import clip
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from model.cbm_models import clip_cbm_orth
from model.domain_grounding import (
    compute_alignment_score,
    build_image_grounded_domain_diffs,
    build_image_grounded_domain_diffs_persample,
    blend_domain_diffs,
)
from cache_utils import get_or_build_feature_cache
from utils import get_domain_text_embs
from prompts.prompt200new import source_text_prompts, target_text_prompts

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
HF_CACHE_DIR = "/data/ai25mtech14009/hf_cache"
CONCEPTS_PATH = "data/Defactify/defactify_concepts.txt"
SEED = 0
EPOCHS = 50
BATCH_SIZE = 64
TEST_FRACTION = 0.2
CAP_PER_CLASS_DOMAIN = 200
MIN_PER_CLASS = 90
TARGET_LABEL_B = 5  # Midjourney v6
TARGET_NAME = "Midjourney v6"
MAX_PROBE = 60  # fixed held-out-from-test cap; every probe size is a prefix of this
PROBE_SIZES = [10, 20, 40, 60]
BLEND_PERSAMPLE_PROBE_SIZE = 20  # matches the original run, for direct comparability
SOURCE_TEMPLATE = "a photo of a {}."
TARGET_TEMPLATE = "a Midjourney-generated image of a {}."

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def first_matching_category(caption):
    cl = caption.lower()
    for cat in COCO_CATEGORIES:
        if cat in cl or f"{cat}s" in cl:
            return cat
    return None


class DefactifyImageDataset(Dataset):
    def __init__(self, ds_dict, entries, preprocess, num_concepts):
        self.ds_dict = ds_dict
        self.entries = entries
        self.preprocess = preprocess
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        split, row_idx, label = self.entries[i]
        image = self.ds_dict[split][row_idx]["Image"].convert("RGB")
        attr_label = torch.tensor([0] * self.num_concepts)
        return self.preprocess(image), label, attr_label


def build_class_index(ds_dict):
    idx_by_label_cat = defaultdict(lambda: defaultdict(list))
    for split_name, ds in ds_dict.items():
        label_b_col = ds["Label_B"]
        captions = ds["Caption"]
        for row_idx, (lbl, cap) in enumerate(zip(label_b_col, captions)):
            if lbl not in (0, TARGET_LABEL_B):
                continue
            tag = first_matching_category(cap)
            if tag is not None:
                idx_by_label_cat[lbl][tag].append((split_name, row_idx))
    return idx_by_label_cat


def select_classes(idx_by_label_cat):
    real_counts = {cat: len(v) for cat, v in idx_by_label_cat[0].items()}
    tgt_counts = {cat: len(v) for cat, v in idx_by_label_cat[TARGET_LABEL_B].items()}
    classes = [
        cat for cat in COCO_CATEGORIES
        if real_counts.get(cat, 0) >= MIN_PER_CLASS and tgt_counts.get(cat, 0) >= MIN_PER_CLASS
    ]
    return classes, real_counts, tgt_counts


def build_entries(idx_by_label_cat, classes, class_to_id, rng):
    """train/source_test are unaffected by probe size (as before). The
    target Midjourney pool is split into a FIXED probe reservoir (first
    MAX_PROBE images/class, itself nested - probe size k uses reservoir[:k])
    and a FIXED target_test (reservoir[MAX_PROBE:]) - the same target_test
    set is used for every probe size tested, so only the probe's own size
    varies across the sweep, not what's evaluated on.

    Returns train_entries, source_test_entries, target_probe_reservoir (list
    of (split, row_idx, class_id), length MAX_PROBE*len(classes), grouped so
    reservoir[:k] gives a valid k-sized probe), target_test_entries, and a
    source_probe_reservoir (same nesting, drawn from the real-photo train
    pool with its own RNG stream)."""
    train_entries, source_test_entries = [], []
    target_probe_reservoir, source_probe_reservoir = [], []
    target_test_entries = []
    per_class_summary = {}
    source_probe_rng = random.Random(SEED + 1)
    for cat in classes:
        cid = class_to_id[cat]

        real_pool = list(idx_by_label_cat[0][cat])
        rng.shuffle(real_pool)
        real_pool = real_pool[:CAP_PER_CLASS_DOMAIN]
        n_test = max(1, round(len(real_pool) * TEST_FRACTION))
        test_part, train_part = real_pool[:n_test], real_pool[n_test:]
        train_entries.extend((s, i, cid) for s, i in train_part)
        source_test_entries.extend((s, i, cid) for s, i in test_part)

        src_probe_pool = list(train_part)
        source_probe_rng.shuffle(src_probe_pool)
        source_probe_reservoir.append([(s, i, cid) for s, i in src_probe_pool[:MAX_PROBE]])

        tgt_pool = list(idx_by_label_cat[TARGET_LABEL_B][cat])
        rng.shuffle(tgt_pool)
        tgt_pool = tgt_pool[:CAP_PER_CLASS_DOMAIN]
        reservoir_part, test_part_tgt = tgt_pool[:MAX_PROBE], tgt_pool[MAX_PROBE:]
        target_probe_reservoir.append([(s, i, cid) for s, i in reservoir_part])
        target_test_entries.extend((s, i, cid) for s, i in test_part_tgt)

        per_class_summary[cat] = {
            "n_source_train": len(train_part), "n_source_test": len(test_part),
            "n_target_probe_reservoir": len(reservoir_part), "n_target_test": len(test_part_tgt),
        }
    return (train_entries, source_test_entries, target_probe_reservoir,
            source_probe_reservoir, target_test_entries, per_class_summary)


def probe_images_by_class(ds_dict, reservoir_by_class, class_names, k, preprocess):
    """reservoir_by_class: list (index-aligned with class_names) of entry
    lists, each >= k long. Returns dict[class_name] -> list of k preprocessed
    image tensors, taking the first k entries (nested prefix)."""
    images_by_class = {}
    for i, c in enumerate(class_names):
        entries = reservoir_by_class[i][:k]
        images_by_class[c] = [
            preprocess(ds_dict[s][row_idx]["Image"].convert("RGB")) for s, row_idx, _cid in entries
        ]
    return images_by_class


def run_condition(args, alpha, label, class_names, concept_names, domain_diffs,
                   train_feats, train_labels, train_attrs,
                   source_feats, source_labels, source_attrs,
                   target_feats, target_labels, target_attrs, device):
    args.alpha = alpha
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = clip_cbm_orth(
        args=args, class_names=class_names, concept_names=concept_names, domain_diffs=domain_diffs
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cls_loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(train_feats, train_labels, train_attrs), batch_size=BATCH_SIZE, shuffle=True)
    source_loader = DataLoader(TensorDataset(source_feats, source_labels, source_attrs), batch_size=BATCH_SIZE, shuffle=False)
    target_loader = DataLoader(TensorDataset(target_feats, target_labels, target_attrs), batch_size=BATCH_SIZE, shuffle=False)

    def eval_loader(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels, _attrs in loader:
                feats, labels = feats.to(device), labels.to(device)
                _, cls_preds, _ = model.forward_cached(feats)
                correct += (cls_preds.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    best_target_acc, best_source_acc, best_epoch = 0.0, 0.0, -1
    final_source_acc, final_target_acc = 0.0, 0.0
    for epoch in range(EPOCHS):
        model.train()
        for feats, labels, attrs in train_loader:
            feats, labels, attrs = feats.to(device), labels.to(device), attrs.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            concept_preds, cls_preds, reg = model.forward_cached(feats)
            cls_loss = cls_loss_fn(cls_preds, labels)
            orth_loss = torch.abs(reg).mean()
            loss = cls_loss + args.alpha * orth_loss
            loss.backward()
            optimizer.step()

        source_acc = eval_loader(source_loader)
        target_acc = eval_loader(target_loader)
        final_source_acc, final_target_acc = source_acc, target_acc
        if target_acc > best_target_acc:
            best_target_acc, best_source_acc, best_epoch = target_acc, source_acc, epoch
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  [{label}] epoch {epoch + 1}/{EPOCHS}: source={source_acc:.2f}% target={target_acc:.2f}% "
                  f"best_target={best_target_acc:.2f}%")

    return {
        "final_source_acc": final_source_acc, "final_target_acc": final_target_acc,
        "best_source_acc": best_source_acc, "best_target_acc": best_target_acc, "best_epoch": best_epoch,
    }


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.epochs, args.batch_size = EPOCHS, BATCH_SIZE
    device = args.device
    torch.manual_seed(SEED)
    random.seed(SEED)

    calib_path = os.path.join(RESULTS_DIR, "component2_alignment_calibration.json")
    with open(calib_path) as f:
        threshold = json.load(f)["calibrated_threshold"]
    print(f"Loaded calibrated threshold: {threshold:.4f}")

    print("Loading Defactify/MS-COCO-AI dataset...")
    ds_dict = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", cache_dir=HF_CACHE_DIR)

    print("Tagging captions against 80 COCO categories, indexing by (Label_B, category)...")
    idx_by_label_cat = build_class_index(ds_dict)
    classes, real_counts, tgt_counts = select_classes(idx_by_label_cat)
    classes = sorted(classes, key=lambda c: -min(real_counts[c], tgt_counts[c]))
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Selected {len(classes)} classes: {classes}")

    rng = random.Random(SEED)
    (train_entries, source_test_entries, target_probe_reservoir, source_probe_reservoir,
     target_test_entries, per_class_summary) = build_entries(idx_by_label_cat, classes, class_to_id, rng)
    print(f"train={len(train_entries)} source_test={len(source_test_entries)} "
          f"target_test(fixed)={len(target_test_entries)} probe_reservoir={MAX_PROBE}/class")

    with open(CONCEPTS_PATH) as f:
        concept_names = [x.rstrip() for x in f.readlines()]

    clip_model, preprocess = clip.load(args.CLIP_type, device=device)

    train_ds = DefactifyImageDataset(ds_dict, train_entries, preprocess, len(concept_names))
    source_test_ds = DefactifyImageDataset(ds_dict, source_test_entries, preprocess, len(concept_names))
    target_test_ds = DefactifyImageDataset(ds_dict, target_test_entries, preprocess, len(concept_names))

    print("Computing text-only domain_diffs against the unmodified 204-entry descriptor pool...")
    text_domain_diffs = []
    for src, tgt in zip(source_text_prompts * len(target_text_prompts), target_text_prompts):
        s_emb, t_emb = get_domain_text_embs(clip_model, [src], [tgt], classes, device)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
        diff = t_emb.float() - s_emb.float()
        diff = diff / diff.norm(dim=-1, keepdim=True)
        text_domain_diffs.append(diff)
    text_domain_diffs = torch.stack(text_domain_diffs, dim=0).to(device)
    print(f"text_domain_diffs shape: {tuple(text_domain_diffs.shape)}")

    cache_prefix = f"Defactify_MJv6_C2var_{args.CLIP_type}".replace("/", "-")
    train_feats, train_labels, train_attrs = get_or_build_feature_cache(
        f"{cache_prefix}_train", train_ds, clip_model, device
    )
    source_feats, source_labels, source_attrs = get_or_build_feature_cache(
        f"{cache_prefix}_source_test", source_test_ds, clip_model, device
    )
    target_feats, target_labels, target_attrs = get_or_build_feature_cache(
        f"{cache_prefix}_target_test", target_test_ds, clip_model, device
    )
    print(f"Cached: train={len(train_feats)} source_test={len(source_feats)} target_test={len(target_feats)}")

    common_args = dict(
        class_names=classes, concept_names=concept_names,
        train_feats=train_feats, train_labels=train_labels, train_attrs=train_attrs,
        source_feats=source_feats, source_labels=source_labels, source_attrs=source_attrs,
        target_feats=target_feats, target_labels=target_labels, target_attrs=target_attrs,
        device=device,
    )

    print("\n=== Training baseline (alpha=0, fixed target_test) ===")
    baseline_res = run_condition(args, 0.0, "baseline", domain_diffs=text_domain_diffs, **common_args)
    print("\n=== Training ddo_text (alpha=1, fixed target_test) ===")
    ddo_text_res = run_condition(args, 1.0, "ddo_text", domain_diffs=text_domain_diffs, **common_args)
    print(f"baseline best_target={baseline_res['best_target_acc']:.2f}%  "
          f"ddo_text best_target={ddo_text_res['best_target_acc']:.2f}%")

    diagnosis_by_probe_size = {}
    grounded_mean_results = {}
    persample_res, blend_res = None, None

    for k in PROBE_SIZES:
        print(f"\n--- Probe size {k}/class ---")
        src_probe = probe_images_by_class(ds_dict, source_probe_reservoir, classes, k, preprocess)
        tgt_probe = probe_images_by_class(ds_dict, target_probe_reservoir, classes, k, preprocess)

        per_class_alignment, mean_alignment = compute_alignment_score(
            clip_model, src_probe, tgt_probe, classes, SOURCE_TEMPLATE, TARGET_TEMPLATE, device,
        )
        trusted = mean_alignment >= threshold
        print(f"  mean alignment (probe={k}): {mean_alignment:.4f} "
              f"({'TRUST text' if trusted else 'fall back'}, threshold {threshold:.4f})")
        diagnosis_by_probe_size[k] = {
            "per_class_alignment": per_class_alignment, "mean_alignment": mean_alignment, "trusted": trusted,
        }

        grounded_mean_diffs = build_image_grounded_domain_diffs(clip_model, src_probe, tgt_probe, classes, device)
        print(f"  Training ddo_grounded_mean_probe{k} (alpha=1)...")
        res = run_condition(args, 1.0, f"ddo_grounded_mean_probe{k}", domain_diffs=grounded_mean_diffs, **common_args)
        grounded_mean_results[k] = res
        print(f"  ddo_grounded_mean_probe{k}: best_target={res['best_target_acc']:.2f}% "
              f"(gain {res['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")

        if k == BLEND_PERSAMPLE_PROBE_SIZE:
            print(f"  Building persample ({k} directions/class) and blend variants...")
            persample_diffs = build_image_grounded_domain_diffs_persample(clip_model, src_probe, tgt_probe, classes, device)
            print(f"  persample_diffs shape: {tuple(persample_diffs.shape)}")
            persample_res = run_condition(args, 1.0, "ddo_grounded_persample", domain_diffs=persample_diffs, **common_args)
            print(f"  ddo_grounded_persample: best_target={persample_res['best_target_acc']:.2f}% "
                  f"(gain {persample_res['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")

            blend_diffs = blend_domain_diffs(text_domain_diffs, grounded_mean_diffs)
            print(f"  blend_diffs shape: {tuple(blend_diffs.shape)}")
            blend_res = run_condition(args, 1.0, "ddo_grounded_blend", domain_diffs=blend_diffs, **common_args)
            print(f"  ddo_grounded_blend: best_target={blend_res['best_target_acc']:.2f}% "
                  f"(gain {blend_res['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")

    print("\n=== Summary (best target acc, gain over baseline) ===")
    print(f"baseline:              {baseline_res['best_target_acc']:.2f}%")
    print(f"ddo_text:               {ddo_text_res['best_target_acc']:.2f}% "
          f"({ddo_text_res['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")
    for k in PROBE_SIZES:
        r = grounded_mean_results[k]
        print(f"ddo_grounded_mean_probe{k}: {r['best_target_acc']:.2f}% "
              f"({r['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")
    print(f"ddo_grounded_persample: {persample_res['best_target_acc']:.2f}% "
          f"({persample_res['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")
    print(f"ddo_grounded_blend:     {blend_res['best_target_acc']:.2f}% "
          f"({blend_res['best_target_acc'] - baseline_res['best_target_acc']:+.2f})")

    results = {
        "clip_type": args.CLIP_type,
        "target_generator": TARGET_NAME,
        "n_classes": len(classes),
        "classes": classes,
        "n_concepts": len(concept_names),
        "max_probe": MAX_PROBE,
        "probe_sizes": PROBE_SIZES,
        "blend_persample_probe_size": BLEND_PERSAMPLE_PROBE_SIZE,
        "n_train": len(train_entries),
        "n_source_test": len(source_test_entries),
        "n_target_test_fixed": len(target_test_entries),
        "per_class_summary": per_class_summary,
        "threshold": threshold,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "baseline": baseline_res,
        "ddo_text": ddo_text_res,
        "diagnosis_by_probe_size": diagnosis_by_probe_size,
        "ddo_grounded_mean_by_probe_size": grounded_mean_results,
        "ddo_grounded_persample": persample_res,
        "ddo_grounded_blend": blend_res,
        "reference_first_run": {
            "note": "component2_defactify_grounding_ddo.py's own run (different, smaller target_test - probe=20 held out, not 60) - not directly comparable numerically, cited for context only",
            "baseline_best_target_acc": 74.27,
            "ddo_text_best_target_acc": 75.03,
            "ddo_grounded_mean_probe20_best_target_acc": 73.81,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component2_defactify_grounding_variants.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
