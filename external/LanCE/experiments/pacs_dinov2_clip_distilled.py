"""PACS: DINOv2 trained concept probe, supervised with CLIP zero-shot scores
as soft pseudo-labels - not real per-image concept labels (PACS doesn't
have any, see planning/07's own CUB-only constraint and the conversation
that led here).

What this can and can't show, stated precisely before the numbers, not
after: DINOv2's probe is trained to imitate CLIP's own photo-domain concept
judgments, so it cannot exceed CLIP's concept-level accuracy by construction
- whatever CLIP gets systematically wrong on PACS, this inherits. The
genuinely open question is different: does a DINOv2-backed classifier,
trained only on CLIP's photo-domain calls, generalize better under domain
shift (art_painting/cartoon/sketch) than CLIP's own zero-shot judgment does
on those same shifted domains - since DINOv2's feature space is a different,
self-supervised, non-text-aligned one, even though the training *targets*
came entirely from CLIP.

Pseudo-label construction: CLIP zero-shot cosine similarity (photo train
split, human-written pacs_concepts.txt - the same 70 real concept names
Option A's CLIP-vs-LLM comparison used, kept fixed here so this experiment
isolates the scoring-mechanism question, not concept-bank content), min-max
normalized per concept across the training set to give a soft [0,1] target
for BCEWithLogitsLoss (a standard distillation approach - a soft target
preserves CLIP's own confidence/uncertainty, rather than picking an
arbitrary hard threshold).

Compares against Option A's existing CLIP zero-shot (human bank, no DDO)
numbers - results/pacs_concept_bank_comparison.json's "human_alpha0.0" key -
without rerunning that pipeline.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import clip
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
META_ROOT = "data/PACS"
CACHE_DIR = "embeddings_cache"
CONCEPT_FILE = "pacs_concepts.txt"  # human bank - fixed, this experiment isolates the scoring mechanism only
SOURCE_DOMAIN = "photo"
TARGET_DOMAINS = ["art_painting", "cartoon", "sketch"]
SEED = 0
PROBE_EPOCHS = 50
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
CLS_EPOCHS = 50
CLS_LR = 1e-4
CLS_BATCH_SIZE = 64
ENCODE_BATCH_SIZE = 128
DINOV2_SHORT_NAME = "dinov2_vitl14"
DINOV2_TIMM_NAME = "vit_large_patch14_dinov2.lvd142m"


class PACSRawDataset(Dataset):
    def __init__(self, data_root, list_path, transform):
        with open(list_path) as f:
            self.annos = f.readlines()
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label = self.annos[idx].strip().split(",")
        image = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
        image = self.transform(image)
        return image, int(cls_label)


def load_clip_cache(domain, split, clip_type):
    cache_path = os.path.join(CACHE_DIR, f"PACS_{clip_type.replace('/', '-')}_{domain}_{split}.pt")
    cached = torch.load(cache_path)
    return cached["features"], cached["labels"]


def build_dinov2_cache(name, dataset, encode_fn, device, batch_size=ENCODE_BATCH_SIZE):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path)
        return cached["features"], cached["labels"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = encode_fn(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({"features": features, "labels": labels}, cache_path)
    return features, labels


def train_concept_probe(train_feats, soft_targets, feat_dim, n_concepts, device):
    head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, n_concepts)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_feats, soft_targets), batch_size=ENCODE_BATCH_SIZE, shuffle=True)
    head.train()
    for epoch in range(PROBE_EPOCHS):
        total_loss = 0.0
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(feats), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{PROBE_EPOCHS}: BCE loss={total_loss / len(train_feats):.4f}")
    head.eval()
    return head


def train_downstream_classifier(concept_train, labels_train, n_concepts, n_classes, device):
    classifier = nn.Sequential(nn.LayerNorm(n_concepts), nn.Linear(n_concepts, n_classes)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=CLS_LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(concept_train, labels_train), batch_size=CLS_BATCH_SIZE, shuffle=True)
    classifier.train()
    for epoch in range(CLS_EPOCHS):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(classifier(feats), labels)
            loss.backward()
            optimizer.step()
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

    data_root = os.path.join(args.data_dir, "PACS", "pacs_raw", "kfold")
    with open(os.path.join(META_ROOT, CONCEPT_FILE)) as f:
        concept_names = [x.rstrip() for x in f.readlines()]
    n_concepts = len(concept_names)
    print(f"n_concepts={n_concepts}")

    # ---- Step 1: CLIP zero-shot scores on photo train -> soft pseudo-labels ----
    print("=== CLIP zero-shot pseudo-labels (photo train) ===")
    clip_model, _ = clip.load(args.CLIP_type, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        concept_tokens = clip.tokenize(concept_names).to(device)
        concept_text_emb = clip_model.encode_text(concept_tokens).float()
        concept_text_emb = concept_text_emb / concept_text_emb.norm(dim=-1, keepdim=True)

    clip_train_feats, class_labels_train = load_clip_cache(SOURCE_DOMAIN, "train", args.CLIP_type)
    with torch.no_grad():
        clip_scores = clip_train_feats.to(device) @ concept_text_emb.T  # (n_train, n_concepts)
        c_min = clip_scores.min(dim=0, keepdim=True).values
        c_max = clip_scores.max(dim=0, keepdim=True).values
        soft_targets = ((clip_scores - c_min) / (c_max - c_min + 1e-8)).cpu()
    print(f"  soft target range: [{soft_targets.min():.3f}, {soft_targets.max():.3f}], "
          f"mean={soft_targets.mean():.3f}")
    del clip_model
    torch.cuda.empty_cache()

    # ---- Step 2: encode PACS (all domains needed) via DINOv2 ----
    print(f"\n=== {DINOV2_SHORT_NAME} ({DINOV2_TIMM_NAME}) ===")
    backbone = timm.create_model(DINOV2_TIMM_NAME, pretrained=True, num_classes=0).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    data_config = timm.data.resolve_data_config(model=backbone)
    transform = timm.data.create_transform(**data_config, is_training=False)
    feat_dim = backbone.num_features
    print(f"  feature dim={feat_dim}")

    all_domains = [SOURCE_DOMAIN] + TARGET_DOMAINS
    dino_feats, dino_labels = {}, {}
    for domain in all_domains:
        splits = ["train", "test"] if domain == SOURCE_DOMAIN else ["test"]
        for split in splits:
            ds = PACSRawDataset(
                os.path.join(data_root, domain),
                os.path.join(META_ROOT, f"pacs_{domain}_{split}.txt"),
                transform,
            )
            feats, labels = build_dinov2_cache(f"pacs_{DINOV2_SHORT_NAME}_{domain}_{split}", ds, backbone, device)
            dino_feats[(domain, split)] = feats
            dino_labels[(domain, split)] = labels
    del backbone
    torch.cuda.empty_cache()

    # sanity: class labels from the DINOv2-side loader must match the CLIP-side cache's own labels/order
    assert torch.equal(dino_labels[(SOURCE_DOMAIN, "train")], class_labels_train), \
        "photo train split order mismatch between CLIP cache and DINOv2 loader"

    # ---- Step 3: train the concept probe against CLIP's soft pseudo-labels ----
    head = train_concept_probe(dino_feats[(SOURCE_DOMAIN, "train")], soft_targets, feat_dim, n_concepts, device)
    concept_vecs = {}
    with torch.no_grad():
        for key, feats in dino_feats.items():
            concept_vecs[key] = torch.sigmoid(head(feats.to(device))).cpu()

    # ---- Step 4: downstream classifier on real class labels (PACS has these) ----
    n_classes = int(class_labels_train.max().item()) + 1
    classifier = train_downstream_classifier(
        concept_vecs[(SOURCE_DOMAIN, "train")], class_labels_train, n_concepts, n_classes, device
    )

    results = {}
    for domain in all_domains:
        acc = eval_accuracy(classifier, concept_vecs[(domain, "test")], dino_labels[(domain, "test")], device)
        results[domain] = acc
        print(f"  {domain}: {acc * 100:.2f}%")

    out = {
        "backbone": DINOV2_SHORT_NAME,
        "concept_file": CONCEPT_FILE,
        "pseudo_label_source": "CLIP zero-shot, photo train, min-max normalized per concept",
        "seed": SEED,
        "n_concepts": n_concepts,
        "results": results,
        "compare_against": "results/pacs_concept_bank_comparison.json, key human_alpha0.0",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pacs_dinov2_clip_distilled.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
