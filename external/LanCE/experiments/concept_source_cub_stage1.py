"""Plan 07, Stage 1 - does a directly-trained concept classifier on a strong
non-CLIP pretrained backbone (DINOv2) produce more accurate concept
activations than CLIP's zero-shot similarity trick?

Concept-level only: no downstream classifier, no DDO (per plan 07 SS2 - once
concept activations stop coming from CLIP similarity, DDO's text-only
domain-shift trick no longer lives in the same space). Just: for each of
CUB's 312 real, human-labeled attributes, which variant's continuous
concept-activation score agrees better with the real per-image label?

Three concept sources compared, all frozen backbones (matches how CLIP is
used everywhere else in this project - linear probe only, no fine-tuning):
  A - CLIP ViT-L/14 zero-shot: cosine(image_embedding, text_embedding(concept))
  B1 - DINOv2 ViT-B/14 + trained linear probe (LayerNorm -> Linear -> sigmoid)
  B2 - DINOv2 ViT-L/14 + trained linear probe (same head)

Metric: per-concept AUROC against CUB's real attribute labels (test split) -
threshold-free, so a cosine similarity and a trained sigmoid probability are
scored the same way with no arbitrary cutoff to pick.
"""
import os
import sys
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import clip
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
META_ROOT = "data/CUB"
CACHE_DIR = "embeddings_cache"
SEED = 0
PROBE_EPOCHS = 50
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
MIN_TEST_POSITIVES = 10  # flag (not drop) concepts with fewer positive test examples than this

DINOV2_BACKBONES = {
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}


class CUBAttrDataset(Dataset):
    """Raw CUB images + real per-image 312-attribute binary labels.

    Unlike data/CUB/cub_data.py's Processed_CUB_Dataset (whose attr_label is
    always a dummy all-zero tensor), this reads real labels from
    CUBpath2attr.pkl so concept-activation predictions can be checked
    against real ground truth. transform is supplied per-backbone since
    CLIP and DINOv2 expect different preprocessing.
    """

    def __init__(self, meta_root, data_root, split, transform):
        fname = "cub_train.txt" if split == "train" else "cub_test.txt"
        with open(os.path.join(meta_root, fname)) as f:
            self.annos = f.readlines()
        with open(os.path.join(meta_root, "CUBpath2attr.pkl"), "rb") as f:
            self.path2attr = pickle.load(f)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label = self.annos[idx].strip().split(",")[0:2]
        image = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
        image = self.transform(image)
        label = int(cls_label) - 1  # labels are 1-indexed in the source files
        attr = torch.tensor(self.path2attr[img_path], dtype=torch.float32)
        return image, label, attr


def build_feature_cache(name, dataset, encode_fn, device, batch_size=BATCH_SIZE):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path)
        return cached["features"], cached["labels"], cached["attrs"]

    # num_workers=0: see cache_utils.py's own note - num_workers>0 silently
    # produced corrupted/duplicated rows on this environment.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats, all_labels, all_attrs = [], [], []
    with torch.no_grad():
        for images, labels, attrs in tqdm(loader, desc=f"Encoding {name}"):
            images = images.to(device)
            feats = encode_fn(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            all_attrs.append(attrs)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    attrs = torch.cat(all_attrs, dim=0)
    torch.save({"features": features, "labels": labels, "attrs": attrs}, cache_path)
    return features, labels, attrs


def train_linear_probe(train_feats, train_attrs, feat_dim, n_concepts, device):
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, n_concepts),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_feats, train_attrs), batch_size=BATCH_SIZE, shuffle=True)

    head.train()
    for epoch in range(PROBE_EPOCHS):
        total_loss = 0.0
        for feats, attrs in loader:
            feats, attrs = feats.to(device), attrs.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(feats)
            loss = loss_fn(logits, attrs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{PROBE_EPOCHS}: BCE loss={total_loss / len(train_feats):.4f}")
    return head


def per_concept_auroc(scores, ground_truth, concept_names):
    """scores, ground_truth: (n_test, n_concepts) numpy arrays."""
    results = {}
    n_skipped = 0
    for c in range(len(concept_names)):
        y_true = ground_truth[:, c]
        if y_true.min() == y_true.max():
            n_skipped += 1
            continue
        auc = roc_auc_score(y_true, scores[:, c])
        n_pos = int(y_true.sum())
        results[concept_names[c]] = {"auroc": auc, "n_test_positives": n_pos}
    return results, n_skipped


def summarize(per_concept, label):
    aucs = np.array([v["auroc"] for v in per_concept.values()])
    print(f"  {label}: mean AUROC={aucs.mean():.4f}  median={np.median(aucs):.4f}  "
          f"n_concepts_scored={len(aucs)}")
    return {"mean_auroc": float(aucs.mean()), "median_auroc": float(np.median(aucs)),
            "n_concepts_scored": int(len(aucs))}


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = os.path.join(args.data_dir, "CUB/CUB_200_2011/images")
    with open(os.path.join(META_ROOT, "cub_concepts.txt")) as f:
        concept_names = [x.rstrip() for x in f.readlines()]
    n_concepts = len(concept_names)
    print(f"n_concepts={n_concepts}")

    all_variant_results = {}
    ground_truth_test = None

    # ---------------- Variant A: CLIP ViT-L/14 zero-shot ----------------
    print("\n=== Variant A: CLIP ViT-L/14 zero-shot ===")
    clip_model, clip_preprocess = clip.load(args.CLIP_type, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_test_ds = CUBAttrDataset(META_ROOT, data_root, "test", clip_preprocess)

    clip_test_feats, _, clip_test_attrs = build_feature_cache(
        "concept_src_clip-vitl14_test", clip_test_ds, clip_model.encode_image, device
    )
    ground_truth_test = clip_test_attrs.numpy()

    with torch.no_grad():
        concept_tokens = clip.tokenize(concept_names).to(device)
        concept_text_emb = clip_model.encode_text(concept_tokens).float()
        concept_text_emb = concept_text_emb / concept_text_emb.norm(dim=-1, keepdim=True)
    clip_scores = (clip_test_feats.to(device) @ concept_text_emb.T).cpu().numpy()

    per_concept_a, skipped_a = per_concept_auroc(clip_scores, ground_truth_test, concept_names)
    all_variant_results["clip_vitl14_zeroshot"] = {
        "per_concept": per_concept_a,
        "n_skipped_constant_label": skipped_a,
        **summarize(per_concept_a, "CLIP ViT-L/14 zero-shot"),
    }
    del clip_model  # free GPU memory before loading DINOv2

    # ---------------- Variant B: DINOv2 backbones, trained linear probe ----------------
    for short_name, timm_name in DINOV2_BACKBONES.items():
        print(f"\n=== Variant B: {short_name} ({timm_name}) + trained linear probe ===")
        backbone = timm.create_model(timm_name, pretrained=True, num_classes=0).to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        data_config = timm.data.resolve_data_config(model=backbone)
        transform = timm.data.create_transform(**data_config, is_training=False)
        feat_dim = backbone.num_features
        print(f"  feature dim={feat_dim}, input size={data_config.get('input_size')}")

        train_ds = CUBAttrDataset(META_ROOT, data_root, "train", transform)
        test_ds = CUBAttrDataset(META_ROOT, data_root, "test", transform)

        train_feats, _, train_attrs = build_feature_cache(
            f"concept_src_{short_name}_train", train_ds, backbone, device
        )
        test_feats, _, test_attrs = build_feature_cache(
            f"concept_src_{short_name}_test", test_ds, backbone, device
        )
        assert np.array_equal(test_attrs.numpy(), ground_truth_test), (
            f"{short_name} test-split attr labels don't match Variant A's - split order mismatch"
        )

        head = train_linear_probe(train_feats, train_attrs, feat_dim, n_concepts, device)
        head.eval()
        with torch.no_grad():
            probe_scores = torch.sigmoid(head(test_feats.to(device))).cpu().numpy()

        per_concept_b, skipped_b = per_concept_auroc(probe_scores, ground_truth_test, concept_names)
        all_variant_results[short_name] = {
            "per_concept": per_concept_b,
            "n_skipped_constant_label": skipped_b,
            **summarize(per_concept_b, short_name),
        }
        del backbone
        torch.cuda.empty_cache()

    # ---------------- Head-to-head: per-concept win counts vs. CLIP zero-shot ----------------
    print("\n=== Head-to-head vs. CLIP zero-shot (shared concepts only) ===")
    baseline_pc = all_variant_results["clip_vitl14_zeroshot"]["per_concept"]
    win_counts = {}
    for short_name in DINOV2_BACKBONES:
        variant_pc = all_variant_results[short_name]["per_concept"]
        shared = set(baseline_pc) & set(variant_pc)
        wins = sum(1 for c in shared if variant_pc[c]["auroc"] > baseline_pc[c]["auroc"])
        losses = sum(1 for c in shared if variant_pc[c]["auroc"] < baseline_pc[c]["auroc"])
        ties = len(shared) - wins - losses
        win_counts[short_name] = {"n_shared_concepts": len(shared), "wins": wins, "losses": losses, "ties": ties}
        print(f"  {short_name} vs CLIP zero-shot: {wins} wins / {losses} losses / {ties} ties "
              f"(of {len(shared)} shared-scored concepts)")

    results = {
        "seed": SEED,
        "n_concepts_total": n_concepts,
        "min_test_positives_flag_threshold": MIN_TEST_POSITIVES,
        "probe_epochs": PROBE_EPOCHS,
        "variants": all_variant_results,
        "head_to_head_vs_clip_zeroshot": win_counts,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "concept_source_cub_stage1.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
