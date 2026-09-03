"""Plan 08 extended to PACS - does an LLM-generated concept bank beat the
human-written one here too, the way it did on CUB (results/llm_concept_bank_
comparison.md, replicated across two independent draws)?

Trains on PACS's photo domain (source), evaluates in-domain (photo test) and
under domain shift (art_painting/cartoon/sketch test) - the PACS analog of
Plan 08's CUB -> CUB-Painting comparison. Same clip_cbm_orth architecture,
same DDO mechanism, same hyperparameters this project has used for PACS
throughout (results/phase_b_domain_il.md: 50 epochs, batch 64, lr 1e-4,
weight_decay 1e-4) - only --concept_file differs between conditions.

Expectation worth stating up front, not just after: Phase B found PACS's
joint/oracle accuracy is already 98.29% with the human concept bank - PACS's
7 broad classes are an easy task for CLIP concept-activation features,
unlike CUB's 200 fine-grained species. There may be much less headroom here
for a concept-bank swap to show a visible difference, purely because of a
ceiling effect, not because the underlying effect isn't real.

No concept-supervision loss term (beta): PACS's attr_label is a dummy
all-zero tensor regardless of which concept bank is loaded (same as CUB),
so this script skips computing it entirely rather than computing-then-
zeroing it - sidesteps any dependency on the two banks having matching
concept counts (see planning/08's CUB write-up for the count bug this
avoids by construction).
"""
import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from data import get_pacs_datasets, PACS_DOMAINS
from model.cbm_models import clip_cbm_orth
from cache_utils import get_or_build_feature_cache

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
SOURCE_DOMAIN = "photo"
TARGET_DOMAINS = ["art_painting", "cartoon", "sketch"]
SEED = 0
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4


def run_condition(args, concept_file, alpha, device):
    torch.manual_seed(SEED)
    datasets, classname2id, concept2id, domain_diffs = get_pacs_datasets(args, concept_file=concept_file)
    model = clip_cbm_orth(
        args=args, class_names=list(classname2id.keys()),
        concept_names=list(concept2id.keys()), domain_diffs=domain_diffs,
    ).to(device)

    cache_prefix = f"PACS_{args.CLIP_type}".replace("/", "-")
    caches = {}
    for domain in PACS_DOMAINS:
        for split in ("train", "test"):
            feats, labels, _ = get_or_build_feature_cache(
                f"{cache_prefix}_{domain}_{split}", datasets[domain][split], model.clip_model, device,
            )
            caches[(domain, split)] = (feats, labels)

    train_feats, train_labels = caches[(SOURCE_DOMAIN, "train")]
    loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, cls_preds, reg_loss = model.forward_cached(feats)
            cls_loss = loss_fn(cls_preds, labels)
            orth_loss = torch.abs(reg_loss).mean() if reg_loss is not None else torch.tensor(0.0, device=device)
            loss = cls_loss + alpha * orth_loss
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{EPOCHS}: cls={cls_loss.item():.4f}  orth={orth_loss.item():.4f}")

    model.eval()
    results = {}
    with torch.no_grad():
        for domain in [SOURCE_DOMAIN] + TARGET_DOMAINS:
            feats, labels = caches[(domain, "test")]
            feats, labels = feats.to(device), labels.to(device)
            _, cls_preds, _ = model.forward_cached(feats)
            acc = (cls_preds.argmax(dim=-1) == labels).float().mean().item()
            results[domain] = acc
    del model
    torch.cuda.empty_cache()
    return results


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = args.device

    all_results = {}
    for bank_name, concept_file in [("human", "pacs_concepts.txt"), ("llm", "pacs_concepts_llm.txt")]:
        for alpha in [0.0, 1.0]:
            key = f"{bank_name}_alpha{alpha}"
            print(f"\n=== {key} (concept_file={concept_file}) ===")
            res = run_condition(args, concept_file, alpha, device)
            for d, a in res.items():
                print(f"  {d}: {a * 100:.2f}%")
            all_results[key] = res

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pacs_concept_bank_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
