"""Plan 07, Stage 3a - vanilla DDO, reattached unmodified, on top of the
DINOv2 concept source.

Correction recorded in plan 07 SS7: DDO's regularizer term
(clip_cbm_orth.forward_cached, model/cbm_models.py:217) is

    regularizer = self.classifier[1:](self.diffs @ self.concept_embeddings.T)

which never touches the image encoder's output at all - it's built purely
from self.diffs (CLIP-text domain-shift vectors, from GPT-3.5-written
domain descriptor prompts) and self.concept_embeddings (CLIP-text
embeddings of the concept names), independent of whatever produces
concept_activations. So it can be reattached to Stage 2's DINOv2 classifier
completely unmodified - same diffs, same concept_embeddings Phase 0 already
used, computed exactly the same way, zero new machinery.

Reuses Stage 2's cached DINOv2 concept-activation vectors (retrains the
same concept probe - cheap, cached backbone features, no re-encoding) and
adds one term to Stage 2's downstream-classifier training loop:

    total_loss = cls_loss + alpha * |classifier(diffs @ concept_embeddings.T)|.mean()

alpha=1, matching Phase 0's own +DDO run. Only Stage 2's alpha=0 (no
regularizer) result needs comparing against - that run already exists
(results/concept_source_cub_stage2_downstream.json) and is not repeated here.
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_source_cub_stage2_downstream import (
    CUBRawDataset, build_feature_cache, train_concept_probe,
    SPLITS, DINOV2_BACKBONES, N_CLASSES, CLS_EPOCHS, CLS_LR, CLS_BATCH_SIZE,
    META_ROOT, RESULTS_DIR,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from data.CUB.cub_data import Processed_CUB_Dataset
from prompts.prompt200new import source_text_prompts, target_text_prompts

SEED = 0
ALPHA = 1.0  # matches Phase 0's own +DDO run (--alpha 1)


def train_downstream_classifier_with_ddo(concept_train, labels_train, n_concepts, diffs, concept_embeddings, device):
    classifier = nn.Sequential(
        nn.LayerNorm(n_concepts),
        nn.Linear(n_concepts, N_CLASSES),
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=CLS_LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(concept_train, labels_train), batch_size=CLS_BATCH_SIZE, shuffle=True)

    # (num_directions, num_classes, feat_dim) @ (feat_dim, n_concepts) -> (num_directions, num_classes, n_concepts)
    # Fixed for the whole run (diffs/concept_embeddings never change); only
    # classifier's weights change per step, so this is recomputed through
    # the classifier every batch, matching train_cached.py's own per-batch
    # regularizer forward.
    ddo_input = (diffs @ concept_embeddings.T).to(device)

    classifier.train()
    for epoch in range(CLS_EPOCHS):
        total_loss, total_cls, total_orth, n = 0.0, 0.0, 0.0, 0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(feats)
            cls_loss = loss_fn(logits, labels)
            # classifier has no Flatten (unlike cbm_models.py's clip_cbm_orth,
            # whose classifier[1:] skips a Flatten this classifier doesn't
            # have) - both LayerNorm and Linear apply directly.
            orth_loss = torch.abs(classifier(ddo_input)).mean()
            loss = cls_loss + ALPHA * orth_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            total_cls += cls_loss.item() * feats.size(0)
            total_orth += orth_loss.item() * feats.size(0)
            n += feats.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{CLS_EPOCHS}: "
                  f"loss={total_loss/n:.4f}  cls={total_cls/n:.4f}  orth={total_orth/n:.4f}")
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

    # ---- domain_diffs, exactly as Phase 0 computed them (default
    # concept_file=cub_concepts.txt, the human-written 312-concept bank -
    # Stage 3a deliberately reuses Phase 0's own concept space here, not
    # DINOv2's, per plan 07 SS7's "vanilla, reattached unmodified" framing) ----
    print("=== Building domain_diffs (CLIP-text, same as Phase 0) ===")
    data_root = os.path.join(args.data_dir, "CUB/CUB_200_2011/images")
    domain_dataset = Processed_CUB_Dataset(
        args, data_root=data_root, split="train", meta_root=META_ROOT,
        attr_name="CUBpath2attr.pkl",
        src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts,
    )
    diffs = domain_dataset.domain_diffs  # (num_directions, num_classes, embed_dim), already on device
    with open(os.path.join(META_ROOT, "cub_concepts.txt")) as f:
        concept_names = [x.rstrip() for x in f.readlines()]
    n_concepts = len(concept_names)
    print(f"domain_diffs shape={tuple(diffs.shape)}  n_concepts={n_concepts}")

    # ---- CLIP text embeddings of the (human-written) concept names, same
    # object Phase 0's clip_cbm_orth.concept_embeddings is ----
    clip_model, _ = clip.load(args.CLIP_type, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        concept_tokens = clip.tokenize(concept_names).to(device)
        concept_embeddings = clip_model.encode_text(concept_tokens).float()
        concept_embeddings = concept_embeddings / concept_embeddings.norm(dim=-1, keepdim=True)
    del clip_model
    torch.cuda.empty_cache()

    with open(os.path.join(META_ROOT, "cub_train.txt")) as f:
        train_annos = f.readlines()
    with open(os.path.join(META_ROOT, "CUBpath2attr.pkl"), "rb") as f:
        path2attr = pickle.load(f)
    train_attrs = torch.tensor(
        [path2attr[line.strip().split(",")[0]] for line in train_annos], dtype=torch.float32
    )

    all_results = {}
    data_root_base = args.data_dir

    for short_name, timm_name in DINOV2_BACKBONES.items():
        print(f"\n=== {short_name} ({timm_name}) + vanilla DDO reattached (alpha={ALPHA}) ===")
        backbone = timm.create_model(timm_name, pretrained=True, num_classes=0).to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        data_config = timm.data.resolve_data_config(model=backbone)
        transform = timm.data.create_transform(**data_config, is_training=False)
        feat_dim = backbone.num_features

        split_feats, labels_by_split = {}, {}
        for split, (list_file, root_suffix) in SPLITS.items():
            cache_name = f"concept_src_{short_name}_{split}"
            ds = CUBRawDataset(
                os.path.join(data_root_base, root_suffix),
                os.path.join(META_ROOT, list_file),
                transform,
            )
            feats, labels = build_feature_cache(cache_name, ds, backbone, device)
            split_feats[split] = feats
            labels_by_split[split] = labels
        del backbone
        torch.cuda.empty_cache()

        head = train_concept_probe(split_feats["train"], train_attrs, feat_dim, n_concepts, device)
        concept_vecs = {}
        with torch.no_grad():
            for split in SPLITS:
                concept_vecs[split] = torch.sigmoid(head(split_feats[split].to(device))).cpu()

        classifier = train_downstream_classifier_with_ddo(
            concept_vecs["train"], labels_by_split["train"], n_concepts, diffs, concept_embeddings, device
        )
        acc_test = eval_accuracy(classifier, concept_vecs["test"], labels_by_split["test"], device)
        acc_shift = eval_accuracy(classifier, concept_vecs["cubp_test"], labels_by_split["cubp_test"], device)
        print(f"  {short_name} +DDO(reattached): in-domain={acc_test*100:.2f}%  CUB-Painting shift={acc_shift*100:.2f}%")
        all_results[short_name] = {"acc_in_domain": acc_test, "acc_domain_shift": acc_shift}

    results = {
        "seed": SEED,
        "alpha": ALPHA,
        "cls_epochs": CLS_EPOCHS,
        "note": "reuses Phase 0's own CLIP-text domain_diffs and concept_embeddings "
                "(human-written 312-concept cub_concepts.txt), unmodified, per plan 07 SS7 Stage 3a",
        "compare_against_no_ddo": "results/concept_source_cub_stage2_downstream.json",
        "compare_against_clip_ddo": {"phase0_cub_painting_ddo_acc": 0.5704},
        "variants": all_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "concept_source_cub_stage3a_ddo_reattached.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
