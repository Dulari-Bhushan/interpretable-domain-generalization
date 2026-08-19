"""Component 3 - a vocabulary that grows and prunes itself, validated on
Defactify (photo -> Midjourney v6), the same domain shift Component 2 and
Phase E2 already measured.

Reruns Component 2's exact Defactify protocol (same tagging heuristic,
same probe draw, same 50-epoch/batch-64/AdamW/lr=1e-4 setup), adding a
fourth condition: +DDO with a GROWN descriptor pool (the original 204
entries plus new Midjourney-specific templates, generated from the
domain's own name and accepted only if they measure as actually connected
to the probe images via Component 2's own alignment-score formula) in
place of the untouched 204-entry pool.

Four conditions total, trained on one consistent split so every number is
directly comparable:
  - baseline      (alpha=0, no DDO)
  - ddo_text      (alpha=1, original 204-entry pool)      -> reproduces E2/C2's +0.68
  - ddo_grounded  (alpha=1, Component 2's image-grounded fallback) -> reproduces C2's result
  - ddo_grown     (alpha=1, original pool + accepted grown templates)

The empirical question this answers: does growing the vocabulary (adding
words the pool is actually missing) recover more of the +6.40-vs-+0.68 gap
Phase E2 found than Component 2's image-grounded fallback did (which made
things worse, not better).
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
from model.domain_grounding import diagnose_and_build_domain_diffs
from model.vocabulary_growth import grow_descriptor_pool, prune_redundant_descriptors
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
PROBE_PER_CLASS = 20  # cheap: a tenth of the 200/class training cap
SOURCE_TEMPLATE = "a photo of a {}."
TARGET_TEMPLATE = "a Midjourney-generated image of a {}."
GROWTH_DOMAIN_NAME = "Midjourney"  # fills CANDIDATE_TEMPLATE_PATTERNS' {domain}
REDUNDANCY_THRESHOLD = 0.97

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
    train_entries, source_test_entries = [], []
    target_probe_entries, target_test_entries = [], []
    per_class_summary = {}
    for cat in classes:
        cid = class_to_id[cat]

        real_pool = list(idx_by_label_cat[0][cat])
        rng.shuffle(real_pool)
        real_pool = real_pool[:CAP_PER_CLASS_DOMAIN]
        n_test = max(1, round(len(real_pool) * TEST_FRACTION))
        test_part, train_part = real_pool[:n_test], real_pool[n_test:]
        train_entries.extend((s, i, cid) for s, i in train_part)
        source_test_entries.extend((s, i, cid) for s, i in test_part)

        tgt_pool = list(idx_by_label_cat[TARGET_LABEL_B][cat])
        rng.shuffle(tgt_pool)
        tgt_pool = tgt_pool[:CAP_PER_CLASS_DOMAIN]
        probe_part = tgt_pool[:PROBE_PER_CLASS]
        test_part_tgt = tgt_pool[PROBE_PER_CLASS:]
        target_probe_entries.extend((s, i, cid) for s, i in probe_part)
        target_test_entries.extend((s, i, cid) for s, i in test_part_tgt)

        per_class_summary[cat] = {
            "n_source_train": len(train_part), "n_source_test": len(test_part),
            "n_target_probe": len(probe_part), "n_target_test": len(test_part_tgt),
        }
    return train_entries, source_test_entries, target_probe_entries, target_test_entries, per_class_summary


def load_images_by_class(ds_dict, entries, class_names, id_to_class, preprocess):
    images_by_class = {c: [] for c in class_names}
    for split, row_idx, cid in entries:
        image = ds_dict[split][row_idx]["Image"].convert("RGB")
        images_by_class[id_to_class[cid]].append(preprocess(image))
    return images_by_class


def text_domain_diffs_for_pool(clip_model, pool, classes, device):
    """Same computation as component2_defactify_grounding_ddo.py's
    text_domain_diffs build, factored out so it can run over any pool
    (original 204, or original+grown)."""
    diffs = []
    for tgt in pool:
        s_emb, t_emb = get_domain_text_embs(clip_model, source_text_prompts, [tgt], classes, device)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
        diff = t_emb.float() - s_emb.float()
        diff = diff / diff.norm(dim=-1, keepdim=True)
        diffs.append(diff)
    return torch.stack(diffs, dim=0).to(device)


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

    source_curve, target_curve = [], []
    best_target_acc, best_source_acc, best_epoch = 0.0, 0.0, -1
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
        source_curve.append(source_acc)
        target_curve.append(target_acc)
        if target_acc > best_target_acc:
            best_target_acc, best_source_acc, best_epoch = target_acc, source_acc, epoch
        print(f"  [{label}] epoch {epoch + 1}/{EPOCHS}: source={source_acc:.2f}% target={target_acc:.2f}% "
              f"best_target={best_target_acc:.2f}%")

    return {
        "source_curve": source_curve, "target_curve": target_curve,
        "final_source_acc": source_curve[-1], "final_target_acc": target_curve[-1],
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
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            threshold = json.load(f)["calibrated_threshold"]
        print(f"Loaded calibrated threshold from {calib_path}: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"No calibration file found at {calib_path} - using uncalibrated default threshold {threshold}")

    print("Loading Defactify/MS-COCO-AI dataset (downloads on first run)...")
    ds_dict = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", cache_dir=HF_CACHE_DIR)

    print("Tagging captions against 80 COCO categories, indexing by (Label_B, category)...")
    idx_by_label_cat = build_class_index(ds_dict)

    classes, real_counts, tgt_counts = select_classes(idx_by_label_cat)
    classes = sorted(classes, key=lambda c: -min(real_counts[c], tgt_counts[c]))
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    print(f"Selected {len(classes)} classes (>= {MIN_PER_CLASS} tagged samples on both sides): {classes}")

    rng = random.Random(SEED)
    train_entries, source_test_entries, target_probe_entries, target_test_entries, per_class_summary = build_entries(
        idx_by_label_cat, classes, class_to_id, rng
    )
    print(f"train={len(train_entries)} source_test={len(source_test_entries)} "
          f"target_probe={len(target_probe_entries)} target_test={len(target_test_entries)}")

    with open(CONCEPTS_PATH) as f:
        concept_names = [x.rstrip() for x in f.readlines()]
    print(f"Concept bank: {len(concept_names)} concepts for {len(classes)} classes")

    clip_model, preprocess = clip.load(args.CLIP_type, device=device)

    train_ds = DefactifyImageDataset(ds_dict, train_entries, preprocess, len(concept_names))
    source_test_ds = DefactifyImageDataset(ds_dict, source_test_entries, preprocess, len(concept_names))
    target_test_ds = DefactifyImageDataset(ds_dict, target_test_entries, preprocess, len(concept_names))

    print("Computing text-only domain_diffs against the ACTUAL unmodified 204-entry frozen descriptor pool...")
    text_domain_diffs = text_domain_diffs_for_pool(clip_model, target_text_prompts, classes, device)
    print(f"text_domain_diffs shape: {tuple(text_domain_diffs.shape)}")

    print(f"\nDrawing probes and running self-diagnosis on a {PROBE_PER_CLASS}/class probe "
          f"({SOURCE_TEMPLATE!r} -> {TARGET_TEMPLATE!r})...")
    source_probe_rng = random.Random(SEED + 1)
    real_pool_by_class = defaultdict(list)
    for s, i, cid in train_entries:
        real_pool_by_class[cid].append((s, i, cid))
    source_probe_entries = []
    for cid, pool in real_pool_by_class.items():
        pool = list(pool)
        source_probe_rng.shuffle(pool)
        source_probe_entries.extend(pool[:PROBE_PER_CLASS])

    source_images_by_class = load_images_by_class(ds_dict, source_probe_entries, classes, id_to_class, preprocess)
    target_images_by_class = load_images_by_class(ds_dict, target_probe_entries, classes, id_to_class, preprocess)

    diagnosis = diagnose_and_build_domain_diffs(
        clip_model, source_images_by_class, target_images_by_class, classes,
        SOURCE_TEMPLATE, TARGET_TEMPLATE, text_domain_diffs, threshold, device,
    )
    print(f"Mean alignment: {diagnosis['mean_alignment']:.4f} (threshold {threshold:.4f}) "
          f"-> {'TRUST text' if diagnosis['trusted'] else 'FALL BACK to image-grounded domain_diffs'}")
    grounded_domain_diffs = diagnosis["domain_diffs"]

    print(f"\nGrowing the descriptor pool for domain '{GROWTH_DOMAIN_NAME}'...")
    grown_pool, accepted, rejected = grow_descriptor_pool(
        clip_model, target_text_prompts, GROWTH_DOMAIN_NAME,
        source_images_by_class, target_images_by_class, classes,
        SOURCE_TEMPLATE, threshold, device,
    )
    print(f"Accepted {len(accepted)}/{len(accepted) + len(rejected)} candidate templates (threshold {threshold:.4f}):")
    for cand, score in sorted(accepted.items(), key=lambda kv: -kv[1]):
        print(f"  ACCEPT ({score:.4f}) {cand!r}")
    for cand, score in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  reject ({score:.4f}) {cand!r}")
    print(f"Grown pool size: {len(target_text_prompts)} -> {len(grown_pool)}")

    print(f"\nPruning grown pool for redundancy (threshold {REDUNDANCY_THRESHOLD})...")
    pruned_pool, dropped = prune_redundant_descriptors(
        clip_model, grown_pool, device, redundancy_threshold=REDUNDANCY_THRESHOLD,
        protected_prefix_len=len(target_text_prompts),
    )
    print(f"Pruned {len(dropped)} redundant entries: {dropped}")
    print(f"Final grown pool size after pruning: {len(pruned_pool)}")

    grown_domain_diffs = text_domain_diffs_for_pool(clip_model, pruned_pool, classes, device)
    print(f"grown_domain_diffs shape: {tuple(grown_domain_diffs.shape)}")

    cache_prefix = f"Defactify_MJv6_C3_{args.CLIP_type}".replace("/", "-")
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

    conditions = [
        ("baseline", 0.0, text_domain_diffs),
        ("ddo_text", 1.0, text_domain_diffs),
        ("ddo_grounded", 1.0, grounded_domain_diffs),
        ("ddo_grown", 1.0, grown_domain_diffs),
    ]
    results_by_condition = {}
    for label, alpha, diffs in conditions:
        print(f"\n=== Training {label} (alpha={alpha}) ===")
        res = run_condition(
            args, alpha, label, classes, concept_names, diffs,
            train_feats, train_labels, train_attrs,
            source_feats, source_labels, source_attrs,
            target_feats, target_labels, target_attrs, device,
        )
        results_by_condition[label] = res
        print(f"{label}: best_target_acc={res['best_target_acc']:.2f}% (epoch {res['best_epoch'] + 1}), "
              f"best_source_acc={res['best_source_acc']:.2f}%")

    baseline_acc = results_by_condition["baseline"]["best_target_acc"]
    gains = {
        label: results_by_condition[label]["best_target_acc"] - baseline_acc
        for label in ("ddo_text", "ddo_grounded", "ddo_grown")
    }
    print(f"\n=== Summary ===")
    print(f"Baseline target acc:        {baseline_acc:.2f}%")
    for label in ("ddo_text", "ddo_grounded", "ddo_grown"):
        print(f"+DDO ({label}) target acc: {results_by_condition[label]['best_target_acc']:.2f}% "
              f"(gain {gains[label]:+.2f})")

    results = {
        "clip_type": args.CLIP_type,
        "target_generator": TARGET_NAME,
        "growth_domain_name": GROWTH_DOMAIN_NAME,
        "n_classes": len(classes),
        "classes": classes,
        "n_concepts": len(concept_names),
        "min_per_class": MIN_PER_CLASS,
        "cap_per_class_domain": CAP_PER_CLASS_DOMAIN,
        "probe_per_class": PROBE_PER_CLASS,
        "n_train": len(train_entries),
        "n_source_test": len(source_test_entries),
        "n_target_probe": len(target_probe_entries),
        "n_target_test": len(target_test_entries),
        "per_class_summary": per_class_summary,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "original_pool_size": len(target_text_prompts),
        "diagnosis": {
            "per_class_alignment": diagnosis["per_class_alignment"],
            "mean_alignment": diagnosis["mean_alignment"],
            "threshold": threshold,
            "trusted": diagnosis["trusted"],
        },
        "growth": {
            "accepted": accepted,
            "rejected": rejected,
            "grown_pool_size_before_pruning": len(grown_pool),
        },
        "pruning": {
            "redundancy_threshold": REDUNDANCY_THRESHOLD,
            "dropped": dropped,
            "final_pool_size": len(pruned_pool),
        },
        "results_by_condition": results_by_condition,
        "gain_points": gains,
        "phase_e2_ddo_gain_points_reference": 0.68,
        "phase0_ddo_gain_points_reference": 6.40,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component3_defactify_growing_vocab.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
