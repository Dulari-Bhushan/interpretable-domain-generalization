"""Component 2 follow-up: second validation domain (results/component2b's
"What's next" #3). Everything so far (component2_defactify_grounding_ddo.py,
component2_defactify_grounding_variants.py) tested one domain shift, photo
-> Midjourney v6. This reruns the same core comparison - baseline, text-only
DDO, and the three probe=20 fallback designs (single mean direction,
per-sample diversity-preserving, blend) - on photo -> DALL-E 3, Phase F3's
most architecturally distinct generator from Midjourney (alignment 0.017
global / 0.023 per-class, the lowest of the 5 generators tested, vs.
Midjourney's 0.096/0.034) and a different underlying generative approach.

Skips the probe-size sweep (10/40/60) deliberately - that instability
finding was already established on Midjourney and doesn't need re-running
per generator; this script exists to check whether the persample/blend fix
generalizes to a second, independently-alignment-measured domain, not to
re-litigate probe size.

Same fixed protocol as component2_defactify_grounding_variants.py: target_test
is fixed (everything beyond the 20-image/class probe reservoir) so baseline
and ddo_text are trained once and compared fairly against every grounded
variant.
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
TARGET_LABEL_B = 4  # DALL-E 3 (per phase_f3_defactify_alignment.py's GENERATOR_INFO)
TARGET_NAME = "DALL-E 3"
PROBE_SIZE = 20
SOURCE_TEMPLATE = "a photo of a {}."
TARGET_TEMPLATE = "a DALL-E 3-generated image of a {}."

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
    target_probe_entries, source_probe_entries = [], []
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
        source_probe_entries.extend((s, i, cid) for s, i in src_probe_pool[:PROBE_SIZE])

        tgt_pool = list(idx_by_label_cat[TARGET_LABEL_B][cat])
        rng.shuffle(tgt_pool)
        tgt_pool = tgt_pool[:CAP_PER_CLASS_DOMAIN]
        probe_part, test_part_tgt = tgt_pool[:PROBE_SIZE], tgt_pool[PROBE_SIZE:]
        target_probe_entries.extend((s, i, cid) for s, i in probe_part)
        target_test_entries.extend((s, i, cid) for s, i in test_part_tgt)

        per_class_summary[cat] = {
            "n_source_train": len(train_part), "n_source_test": len(test_part),
            "n_target_probe": len(probe_part), "n_target_test": len(test_part_tgt),
        }
    return (train_entries, source_test_entries, target_probe_entries, source_probe_entries,
            target_test_entries, per_class_summary)


def load_images_by_class(ds_dict, entries, class_names, id_to_class, preprocess):
    images_by_class = {c: [] for c in class_names}
    for split, row_idx, cid in entries:
        image = ds_dict[split][row_idx]["Image"].convert("RGB")
        images_by_class[id_to_class[cid]].append(preprocess(image))
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
        if target_acc > best_target_acc:
            best_target_acc, best_source_acc, best_epoch = target_acc, source_acc, epoch
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  [{label}] epoch {epoch + 1}/{EPOCHS}: source={source_acc:.2f}% target={target_acc:.2f}% "
                  f"best_target={best_target_acc:.2f}%")

    return {"best_source_acc": best_source_acc, "best_target_acc": best_target_acc, "best_epoch": best_epoch}


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

    print(f"Loading Defactify/MS-COCO-AI dataset, target generator={TARGET_NAME} (Label_B={TARGET_LABEL_B})...")
    ds_dict = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", cache_dir=HF_CACHE_DIR)

    idx_by_label_cat = build_class_index(ds_dict)
    classes, real_counts, tgt_counts = select_classes(idx_by_label_cat)
    classes = sorted(classes, key=lambda c: -min(real_counts[c], tgt_counts[c]))
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    print(f"Selected {len(classes)} classes: {classes}")

    rng = random.Random(SEED)
    (train_entries, source_test_entries, target_probe_entries, source_probe_entries,
     target_test_entries, per_class_summary) = build_entries(idx_by_label_cat, classes, class_to_id, rng)
    print(f"train={len(train_entries)} source_test={len(source_test_entries)} "
          f"target_probe={len(target_probe_entries)} target_test={len(target_test_entries)}")

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

    cache_prefix = f"Defactify_DALLE3_C2d_{args.CLIP_type}".replace("/", "-")
    train_feats, train_labels, train_attrs = get_or_build_feature_cache(f"{cache_prefix}_train", train_ds, clip_model, device)
    source_feats, source_labels, source_attrs = get_or_build_feature_cache(f"{cache_prefix}_source_test", source_test_ds, clip_model, device)
    target_feats, target_labels, target_attrs = get_or_build_feature_cache(f"{cache_prefix}_target_test", target_test_ds, clip_model, device)
    print(f"Cached: train={len(train_feats)} source_test={len(source_feats)} target_test={len(target_feats)}")

    source_images_by_class = load_images_by_class(ds_dict, source_probe_entries, classes, id_to_class, preprocess)
    target_images_by_class = load_images_by_class(ds_dict, target_probe_entries, classes, id_to_class, preprocess)

    per_class_alignment, mean_alignment = compute_alignment_score(
        clip_model, source_images_by_class, target_images_by_class, classes, SOURCE_TEMPLATE, TARGET_TEMPLATE, device,
    )
    trusted = mean_alignment >= threshold
    print(f"Diagnosis: mean alignment={mean_alignment:.4f} (threshold {threshold:.4f}) -> "
          f"{'TRUST text' if trusted else 'fall back'}")

    grounded_mean_diffs = build_image_grounded_domain_diffs(clip_model, source_images_by_class, target_images_by_class, classes, device)
    persample_diffs = build_image_grounded_domain_diffs_persample(clip_model, source_images_by_class, target_images_by_class, classes, device)
    blend_diffs = blend_domain_diffs(text_domain_diffs, grounded_mean_diffs)

    common_args = dict(
        class_names=classes, concept_names=concept_names,
        train_feats=train_feats, train_labels=train_labels, train_attrs=train_attrs,
        source_feats=source_feats, source_labels=source_labels, source_attrs=source_attrs,
        target_feats=target_feats, target_labels=target_labels, target_attrs=target_attrs,
        device=device,
    )

    conditions = [
        ("baseline", 0.0, text_domain_diffs),
        ("ddo_text", 1.0, text_domain_diffs),
        ("ddo_grounded_mean", 1.0, grounded_mean_diffs),
        ("ddo_grounded_persample", 1.0, persample_diffs),
        ("ddo_grounded_blend", 1.0, blend_diffs),
    ]
    results_by_condition = {}
    for label, alpha, diffs in conditions:
        print(f"\n=== Training {label} (alpha={alpha}) ===")
        res = run_condition(args, alpha, label, domain_diffs=diffs, **common_args)
        results_by_condition[label] = res
        gain = res["best_target_acc"] - results_by_condition["baseline"]["best_target_acc"]
        print(f"{label}: best_target_acc={res['best_target_acc']:.2f}% (gain {gain:+.2f})")

    baseline_acc = results_by_condition["baseline"]["best_target_acc"]
    print("\n=== Summary ===")
    for label, _alpha, _diffs in conditions:
        acc = results_by_condition[label]["best_target_acc"]
        print(f"{label}: {acc:.2f}% (gain {acc - baseline_acc:+.2f})")

    results = {
        "clip_type": args.CLIP_type,
        "target_generator": TARGET_NAME,
        "target_label_b": TARGET_LABEL_B,
        "n_classes": len(classes),
        "classes": classes,
        "n_concepts": len(concept_names),
        "probe_size": PROBE_SIZE,
        "n_train": len(train_entries),
        "n_source_test": len(source_test_entries),
        "n_target_test": len(target_test_entries),
        "per_class_summary": per_class_summary,
        "threshold": threshold,
        "diagnosis": {"per_class_alignment": per_class_alignment, "mean_alignment": mean_alignment, "trusted": trusted},
        "training_seed": args.seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "results_by_condition": results_by_condition,
        "phase_f3_reference_alignment": {"global": 0.017, "per_class": 0.023},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component2d_dalle3_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
