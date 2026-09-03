"""Plan 07, Stage 2 - does Stage 1's more-accurate concept-activation vector
(DINOv2 linear probe) translate into better final classification accuracy
than CLIP zero-shot concept activations, both in-domain (CUB test) and under
a real domain shift (CUB-Painting)?

Stage 1 (results/concept_source_cub_stage1.json) already answered the
concept-level question decisively: both DINOv2 sizes beat CLIP zero-shot on
mean per-concept AUROC against CUB's real 312 attributes (0.777/0.772 vs.
0.647), clearing plan 07 SS3's continuation gate. This stage runs the
downstream half.

No DDO (per plan 07 SS2/SS4): once concept activations stop coming from CLIP
similarity, DDO's text-only domain-shift regularizer no longer lives in the
same space, so it's dropped for all three variants here - a fair plain-
classification comparison, matching Phase 0's own baseline (no-DDO) protocol
(results/phase0_cub_reproduction.md) rather than inventing a new one.

Classifier: LayerNorm(n_concepts) -> Linear(n_concepts, 200), matching
clip_cbm's own downstream head (model/cbm_models.py's clip_cbm /
clip_cbm_orth classes) and trained with Phase 0's own hyperparameters
(50 epochs, batch size 64, Adam, lr 1e-4) so the resulting accuracy numbers
are stated the same way as every other accuracy result in this project.

Reuses Stage 1's cached DINOv2 features (embeddings_cache/concept_src_*)
where they exist. Adds: CLIP train-split + CUB-Painting embeddings for all
three variants (CLIP's CUB-Painting embeddings were already cached by Phase
0 - CUB_ViT-L-14_target_test.pt - reused directly rather than recomputed).
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
PROBE_EPOCHS = 50       # concept probe - same as stage 1
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
CLS_EPOCHS = 50         # downstream classifier - matches Phase 0's protocol
CLS_LR = 1e-4
CLS_BATCH_SIZE = 64     # matches Phase 0's train_cached.py
ENCODE_BATCH_SIZE = 128
N_CLASSES = 200

DINOV2_BACKBONES = {
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}

SPLITS = {
    # split_key: (list_file, data_root_suffix)
    "train": ("cub_train.txt", "CUB/CUB_200_2011/images"),
    "test": ("cub_test.txt", "CUB/CUB_200_2011/images"),
    "cubp_test": ("cubp_test.txt", "CUB/CUB-200-Painting"),
}


class CUBRawDataset(Dataset):
    """Raw image + class label only (no attributes needed for Stage 2's
    downstream classifier - it consumes concept-activation vectors, not
    attribute labels directly). Works for all three splits: cub_train.txt,
    cub_test.txt and cubp_test.txt all share the same
    "img_path,cls_label,..." format."""

    def __init__(self, data_root, list_path, transform):
        with open(list_path) as f:
            self.annos = f.readlines()
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label = self.annos[idx].strip().split(",")[0:2]
        image = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
        image = self.transform(image)
        label = int(cls_label) - 1  # 1-indexed in source files
        return image, label


def build_feature_cache(name, dataset, encode_fn, device, batch_size=ENCODE_BATCH_SIZE):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path)
        return cached["features"], cached["labels"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Encoding {name}"):
            images = images.to(device)
            feats = encode_fn(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({"features": features, "labels": labels}, cache_path)
    return features, labels


def load_phase0_clip_cache(name):
    """Phase 0 already cached CLIP ViT-L/14 embeddings for all three CUB
    splits (CUB_ViT-L-14_{train,source_test,target_test}.pt) with a
    {"features","labels","attr_labels"} schema. Reused as-is - no need to
    re-encode ~15K images that were already encoded months ago."""
    cached = torch.load(os.path.join(CACHE_DIR, f"{name}.pt"))
    return cached["features"], cached["labels"]


def train_concept_probe(train_feats, train_attrs, feat_dim, n_concepts, device):
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, n_concepts),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_feats, train_attrs), batch_size=ENCODE_BATCH_SIZE, shuffle=True)

    head.train()
    for epoch in range(PROBE_EPOCHS):
        for feats, attrs in loader:
            feats, attrs = feats.to(device), attrs.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(feats), attrs)
            loss.backward()
            optimizer.step()
    head.eval()
    return head


def train_downstream_classifier(concept_train, labels_train, n_concepts, device):
    classifier = nn.Sequential(
        nn.LayerNorm(n_concepts),
        nn.Linear(n_concepts, N_CLASSES),
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=CLS_LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(concept_train, labels_train), batch_size=CLS_BATCH_SIZE, shuffle=True)

    classifier.train()
    for epoch in range(CLS_EPOCHS):
        total_loss, n = 0.0, 0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            n += feats.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{CLS_EPOCHS}: CE loss={total_loss / n:.4f}")
    classifier.eval()
    return classifier


@torch.no_grad()
def eval_accuracy(classifier, concept_feats, labels, device):
    logits = classifier(concept_feats.to(device))
    preds = logits.argmax(dim=-1).cpu()
    return float((preds == labels).float().mean().item())


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = args.data_dir  # "./data" - SPLITS suffixes already include "CUB/..."
    with open(os.path.join(META_ROOT, "cub_concepts.txt")) as f:
        concept_names = [x.rstrip() for x in f.readlines()]
    n_concepts = len(concept_names)
    with open(os.path.join(META_ROOT, "CUBpath2attr.pkl"), "rb") as f:
        path2attr = pickle.load(f)
    with open(os.path.join(META_ROOT, "cub_train.txt")) as f:
        train_annos = f.readlines()
    train_attrs = torch.tensor(
        [path2attr[line.strip().split(",")[0]] for line in train_annos], dtype=torch.float32
    )
    print(f"n_concepts={n_concepts}  n_train={len(train_annos)}")

    all_variant_results = {}
    labels_by_split = {}

    # ---------------- Variant A: CLIP ViT-L/14 zero-shot ----------------
    print("\n=== Variant A: CLIP ViT-L/14 zero-shot ===")
    clip_model, clip_preprocess = clip.load(args.CLIP_type, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        concept_tokens = clip.tokenize(concept_names).to(device)
        concept_text_emb = clip_model.encode_text(concept_tokens).float()
        concept_text_emb = concept_text_emb / concept_text_emb.norm(dim=-1, keepdim=True)

    clip_feats, clip_labels = {}, {}
    # Phase 0 already cached these three CLIP splits - reuse directly.
    clip_feats["train"], clip_labels["train"] = load_phase0_clip_cache("CUB_ViT-L-14_train")
    clip_feats["test"], clip_labels["test"] = load_phase0_clip_cache("CUB_ViT-L-14_source_test")
    clip_feats["cubp_test"], clip_labels["cubp_test"] = load_phase0_clip_cache("CUB_ViT-L-14_target_test")
    del clip_model
    torch.cuda.empty_cache()

    concept_vecs_a = {}
    for split in SPLITS:
        feats = clip_feats[split].to(device)
        concept_vecs_a[split] = (feats @ concept_text_emb.T).cpu()
        labels_by_split[split] = clip_labels[split]

    classifier_a = train_downstream_classifier(
        concept_vecs_a["train"], labels_by_split["train"], n_concepts, device
    )
    acc_a_test = eval_accuracy(classifier_a, concept_vecs_a["test"], labels_by_split["test"], device)
    acc_a_shift = eval_accuracy(classifier_a, concept_vecs_a["cubp_test"], labels_by_split["cubp_test"], device)
    print(f"  CLIP ViT-L/14 zero-shot: in-domain={acc_a_test*100:.2f}%  CUB-Painting shift={acc_a_shift*100:.2f}%")
    all_variant_results["clip_vitl14_zeroshot"] = {
        "acc_in_domain": acc_a_test, "acc_domain_shift": acc_a_shift,
    }

    # ---------------- Variant B: DINOv2 backbones, trained probe + classifier ----------------
    for short_name, timm_name in DINOV2_BACKBONES.items():
        print(f"\n=== Variant B: {short_name} ({timm_name}) ===")
        backbone = timm.create_model(timm_name, pretrained=True, num_classes=0).to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        data_config = timm.data.resolve_data_config(model=backbone)
        transform = timm.data.create_transform(**data_config, is_training=False)
        feat_dim = backbone.num_features

        split_feats = {}
        for split, (list_file, root_suffix) in SPLITS.items():
            # Stage 1 already cached train/test for both DINOv2 sizes under
            # these exact names; cubp_test is new.
            cache_name = f"concept_src_{short_name}_{split}"
            ds = CUBRawDataset(
                os.path.join(data_root, root_suffix),
                os.path.join(META_ROOT, list_file),
                transform,
            )
            feats, labels = build_feature_cache(cache_name, ds, backbone, device)
            split_feats[split] = feats
            labels_by_split[split] = labels

        head = train_concept_probe(split_feats["train"], train_attrs, feat_dim, n_concepts, device)
        concept_vecs_b = {}
        with torch.no_grad():
            for split in SPLITS:
                concept_vecs_b[split] = torch.sigmoid(head(split_feats[split].to(device))).cpu()

        classifier_b = train_downstream_classifier(
            concept_vecs_b["train"], labels_by_split["train"], n_concepts, device
        )
        acc_b_test = eval_accuracy(classifier_b, concept_vecs_b["test"], labels_by_split["test"], device)
        acc_b_shift = eval_accuracy(classifier_b, concept_vecs_b["cubp_test"], labels_by_split["cubp_test"], device)
        print(f"  {short_name}: in-domain={acc_b_test*100:.2f}%  CUB-Painting shift={acc_b_shift*100:.2f}%")
        all_variant_results[short_name] = {
            "acc_in_domain": acc_b_test, "acc_domain_shift": acc_b_shift,
        }
        del backbone
        torch.cuda.empty_cache()

    results = {
        "seed": SEED,
        "cls_epochs": CLS_EPOCHS,
        "cls_lr": CLS_LR,
        "cls_batch_size": CLS_BATCH_SIZE,
        "n_classes": N_CLASSES,
        "phase0_baseline_domain_shift_acc": 0.5064,  # results/phase0_cub_reproduction.md, alpha=0 - CUB-Painting (target), not in-domain
        "variants": all_variant_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "concept_source_cub_stage2_downstream.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
