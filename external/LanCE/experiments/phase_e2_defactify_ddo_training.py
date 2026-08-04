"""Phase E2 - real baseline-vs-+DDO training run, real photos -> a genuinely
post-cutoff AI generator (Pillar 2).

Phase F3 measured an *alignment score* (a representation-level proxy) for
real-photo-vs-Midjourney and found it near zero (0.037-0.05) - but explicitly
flagged as a limitation that no actual training run was attempted, so the
alignment score's translation into real trained-model accuracy loss was
never directly measured. Phase E1 separately showed LanCE's frozen 204-
descriptor pool contains zero terms for any AI-generated-image domain.

This phase closes both gaps at once: an actual CLIP-CBM baseline (alpha=0)
vs +DDO (alpha=1) training run, source domain = real MS COCO photos, target
domain (OOD test only, never trained on) = Midjourney v6 images of the same
classes - using the EXACT unmodified 204-entry descriptor pool every other
phase trains against (no Midjourney-specific descriptor added; that's the
whole point - this reproduces what actually happens when a model trained
today meets a domain nobody curated for). Same protocol as Phase 0's
CUB->CUB-Painting reproduction (50 epochs, batch 64, lr 1e-4, AdamW,
weight_decay 1e-4) so the DDO gain here is directly comparable to Phase 0's
own measured +6.40-point gain on a domain (painting) the descriptor pool
DOES cover well.

Data: Rajarshi-Roy-research/Defactify_Image_Dataset (HF, cached locally at
C:/hfc from Phase F3's run). Real photos = Label_B==0, Midjourney v6 =
Label_B==5. Captions are keyword-tagged against the 80 standard COCO
categories (same heuristic Phase F3 used) to get class labels; 19
categories with >=90 tagged samples on both the real and Midjourney side
(across all 3 dataset splits combined) were kept, capped at 200
images/class/domain for a balanced, PACS-scale benchmark.
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
from cache_utils import get_or_build_feature_cache
from utils import get_domain_text_embs
from prompts.prompt200new import source_text_prompts, target_text_prompts

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
HF_CACHE_DIR = "C:/hfc"
CONCEPTS_PATH = "data/Defactify/defactify_concepts.txt"
SEED = 0
EPOCHS = 50
BATCH_SIZE = 64
TEST_FRACTION = 0.2
CAP_PER_CLASS_DOMAIN = 200
MIN_PER_CLASS = 90
TARGET_LABEL_B = 5  # Midjourney v6
TARGET_NAME = "Midjourney v6"

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
    """entries: list of (split_name, row_idx, class_id). Loads lazily from
    the already-loaded HF DatasetDict, applies CLIP's preprocess, and
    returns the (image_tensor, label, attr_label) triple cache_utils/
    clip_cbm_orth expect. attr_label is an unused zero placeholder (beta=0
    throughout this project, same pattern as every other dataset loader)."""

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
    """Returns idx_by_label_cat[label_b][category] = [(split, row_idx), ...]"""
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
    """Returns (train_entries, source_test_entries, target_test_entries), each
    a list of (split, row_idx, class_id), and a per-class count summary."""
    train_entries, source_test_entries, target_test_entries = [], [], []
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
        target_test_entries.extend((s, i, cid) for s, i in tgt_pool)

        per_class_summary[cat] = {
            "n_source_train": len(train_part), "n_source_test": len(test_part),
            "n_target_test": len(tgt_pool),
        }
    return train_entries, source_test_entries, target_test_entries, per_class_summary


def run_condition(args, alpha, class_names, concept_names, domain_diffs,
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
        print(f"  [alpha={alpha}] epoch {epoch + 1}/{EPOCHS}: source={source_acc:.2f}% target={target_acc:.2f}% "
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

    print("Loading Defactify/MS-COCO-AI dataset (cached locally)...")
    ds_dict = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", cache_dir=HF_CACHE_DIR)

    print("Tagging captions against 80 COCO categories, indexing by (Label_B, category)...")
    idx_by_label_cat = build_class_index(ds_dict)

    classes, real_counts, tgt_counts = select_classes(idx_by_label_cat)
    classes = sorted(classes, key=lambda c: -min(real_counts[c], tgt_counts[c]))
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Selected {len(classes)} classes (>= {MIN_PER_CLASS} tagged samples on both sides): {classes}")

    rng = random.Random(SEED)
    train_entries, source_test_entries, target_test_entries, per_class_summary = build_entries(
        idx_by_label_cat, classes, class_to_id, rng
    )
    print(f"train={len(train_entries)} source_test={len(source_test_entries)} target_test={len(target_test_entries)}")

    with open(CONCEPTS_PATH) as f:
        concept_names = [x.rstrip() for x in f.readlines()]
    print(f"Concept bank: {len(concept_names)} concepts for {len(classes)} classes")

    clip_model, preprocess = clip.load(args.CLIP_type, device=device)

    train_ds = DefactifyImageDataset(ds_dict, train_entries, preprocess, len(concept_names))
    source_test_ds = DefactifyImageDataset(ds_dict, source_test_entries, preprocess, len(concept_names))
    target_test_ds = DefactifyImageDataset(ds_dict, target_test_entries, preprocess, len(concept_names))

    print("Computing domain_diffs against the ACTUAL unmodified 204-entry frozen descriptor pool...")
    domain_diffs = []
    for src, tgt in zip(source_text_prompts * len(target_text_prompts), target_text_prompts):
        s_emb, t_emb = get_domain_text_embs(clip_model, [src], [tgt], classes, device)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
        diff = t_emb.float() - s_emb.float()
        diff = diff / diff.norm(dim=-1, keepdim=True)
        domain_diffs.append(diff)
    domain_diffs = torch.stack(domain_diffs, dim=0).to(device)
    print(f"domain_diffs shape: {tuple(domain_diffs.shape)}")

    cache_prefix = f"Defactify_MJv6_{args.CLIP_type}".replace("/", "-")
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

    results_by_alpha = {}
    for alpha in [0.0, 1.0]:
        label = "baseline" if alpha == 0.0 else "ddo"
        print(f"\n=== Training {label} (alpha={alpha}) ===")
        res = run_condition(
            args, alpha, classes, concept_names, domain_diffs,
            train_feats, train_labels, train_attrs,
            source_feats, source_labels, source_attrs,
            target_feats, target_labels, target_attrs, device,
        )
        results_by_alpha[label] = res
        print(f"{label}: best_target_acc={res['best_target_acc']:.2f}% (epoch {res['best_epoch'] + 1}), "
              f"best_source_acc={res['best_source_acc']:.2f}%")

    ddo_gain = results_by_alpha["ddo"]["best_target_acc"] - results_by_alpha["baseline"]["best_target_acc"]
    print(f"\n=== Summary ===")
    print(f"Baseline target acc: {results_by_alpha['baseline']['best_target_acc']:.2f}%")
    print(f"+DDO target acc:     {results_by_alpha['ddo']['best_target_acc']:.2f}%")
    print(f"DDO gain: {ddo_gain:+.2f} points (Phase 0's CUB->Painting gain was +6.40 points)")

    results = {
        "clip_type": args.CLIP_type,
        "target_generator": TARGET_NAME,
        "n_classes": len(classes),
        "classes": classes,
        "n_concepts": len(concept_names),
        "min_per_class": MIN_PER_CLASS,
        "cap_per_class_domain": CAP_PER_CLASS_DOMAIN,
        "n_train": len(train_entries),
        "n_source_test": len(source_test_entries),
        "n_target_test": len(target_test_entries),
        "per_class_summary": per_class_summary,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "descriptor_pool_size": len(target_text_prompts),
        "results_by_condition": results_by_alpha,
        "ddo_gain_points": ddo_gain,
        "phase0_ddo_gain_points_reference": 6.40,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_e2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
