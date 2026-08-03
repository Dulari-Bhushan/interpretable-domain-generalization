"""Phase F2 - Concept-activation ceiling test on EuroSAT (Pillar 2, conditional
on F1 justifying it - F1's mean alignment score was 0.32, far below the
paper's own 0.90-0.99 range for domains it handles well, so it does).

NOT a domain-generalization test - EuroSAT's classes aren't in the same
label space as PACS/Office-Home/CUB, and there's no train/test domain
shift here at all (train and test are both EuroSAT satellite imagery,
80/20 split). It's a precondition check: does concept-based classification
work *at all* in a modality CLIP aligns poorly with (per F1), even though
F1 also showed the *visual* information is there (our own zero-shot
accuracy, 64.05%, plausible-but-limited, nowhere near the 98.1%
linear-probe ceiling OpenAI reports on the identical visual features).

Trains a plain CLIP-CBM - same clip_cbm_orth architecture used throughout
this project, alpha=0 so the DDO term contributes nothing to the loss ("no
DDO", per the plan) - on EuroSAT's own 10 classes, in-distribution, and
compares its ceiling accuracy to F1's two anchors: the ~60-64% zero-shot
ceiling and the (not reproduced here, cited from OpenAI's CLIP paper)
98.1% linear-probe ceiling. If CBM accuracy tracks near the zero-shot
number rather than the linear-probe number, that shows concept-bottleneck
classification inherits CLIP's alignment weakness even where the visual
information is demonstrably present - a ceiling no continual-learning fix
to the classifier can lift, because the ceiling is set by the frozen
backbone underneath.
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
from torchvision.datasets import EuroSAT

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
EUROSAT_ROOT = "data/EuroSAT_raw"
CONCEPTS_PATH = "data/EuroSAT/eurosat_concepts.txt"
TEST_FRACTION = 0.2
SEED = 0
EPOCHS = 50
BATCH_SIZE = 64

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


class EuroSATWrapped(Dataset):
    """Adapts torchvision's EuroSAT (ImageFolder) to the (image, label,
    attr_label) interface clip_cbm_orth/cache_utils expect, restricted to
    a given index subset (train or test split)."""

    def __init__(self, base_dataset, indices, num_concepts):
        self.base = base_dataset
        self.indices = indices
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        attr_label = torch.tensor([0] * self.num_concepts)
        return image, label, attr_label


def stratified_split(base_dataset, test_fraction=TEST_FRACTION, seed=SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, (_path, label) in enumerate(base_dataset.samples):
        by_class[label].append(i)
    train_idx, test_idx = [], []
    for label, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_test = max(1, round(len(idxs) * test_fraction))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    return train_idx, test_idx


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.alpha, args.beta = 0.0, 0.0  # "plain CBM, no DDO"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    clip_model, preprocess = clip.load(args.CLIP_type, device=args.device)

    base_ds = EuroSAT(root=EUROSAT_ROOT, download=True, transform=preprocess)
    class_names = base_ds.classes
    display_names = [CLASS_DISPLAY_NAMES[c] for c in class_names]

    with open(CONCEPTS_PATH) as f:
        concept_names = [x.rstrip() for x in f.readlines()]

    train_idx, test_idx = stratified_split(base_ds)
    train_ds = EuroSATWrapped(base_ds, train_idx, len(concept_names))
    test_ds = EuroSATWrapped(base_ds, test_idx, len(concept_names))
    print(f"train={len(train_ds)} test={len(test_ds)} concepts={len(concept_names)}")

    # domain_diffs: unused numerically since alpha=0 zeroes its loss
    # contribution entirely, but clip_cbm_orth's forward still needs a
    # (n_descriptors, n_classes, feat_dim) tensor to run - computed via the
    # same source/target text-prompt machinery as every other phase, purely
    # as a harmless placeholder here.
    print("Computing placeholder domain_diffs (unused, alpha=0)...")
    domain_diffs = []
    for src, tgt in zip(source_text_prompts * len(target_text_prompts), target_text_prompts):
        s_emb, t_emb = get_domain_text_embs(clip_model, [src], [tgt], display_names, args.device)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
        diff = t_emb.float() - s_emb.float()
        diff = diff / diff.norm(dim=-1, keepdim=True)
        domain_diffs.append(diff)
    domain_diffs = torch.stack(domain_diffs, dim=0).to(args.device)

    model = clip_cbm_orth(
        args=args, class_names=display_names, concept_names=concept_names, domain_diffs=domain_diffs
    ).to(args.device)

    cache_prefix = f"EuroSAT_{args.CLIP_type}".replace("/", "-")
    train_feats, train_labels, train_attrs = get_or_build_feature_cache(
        f"{cache_prefix}_train", train_ds, clip_model, args.device
    )
    test_feats, test_labels, test_attrs = get_or_build_feature_cache(
        f"{cache_prefix}_test", test_ds, clip_model, args.device
    )

    train_loader = DataLoader(TensorDataset(train_feats, train_labels, train_attrs), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_feats, test_labels, test_attrs), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cls_loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    acc_curve = []
    for epoch in range(EPOCHS):
        model.train()
        for feats, labels, _attrs in train_loader:
            feats, labels = feats.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            _, cls_preds, reg = model.forward_cached(feats)
            loss = cls_loss_fn(cls_preds, labels) + args.alpha * torch.abs(reg).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels, _attrs in test_loader:
                feats, labels = feats.to(args.device), labels.to(args.device)
                _, cls_preds, _ = model.forward_cached(feats)
                correct += (cls_preds.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)
        acc_curve.append(acc)
        print(f"epoch {epoch + 1}/{EPOCHS}: test_acc={acc:.2f}% best={best_acc:.2f}%")

    # F1's anchors, for direct comparison
    f1_path = os.path.join(RESULTS_DIR, "phase_f1_results.json")
    f1_results = json.load(open(f1_path)) if os.path.exists(f1_path) else None

    results = {
        "clip_type": args.CLIP_type,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_concepts": len(concept_names),
        "epochs": EPOCHS,
        "final_test_acc": acc_curve[-1],
        "best_test_acc": best_acc,
        "acc_curve": acc_curve,
        "anchors": {
            "our_zeroshot_acc": f1_results["our_zeroshot_acc"] if f1_results else None,
            "paper_zeroshot_acc_336px": f1_results["paper_zeroshot_acc_336px"] if f1_results else 59.6,
            "paper_linear_probe_acc_336px": f1_results["paper_linear_probe_acc_336px"] if f1_results else 98.1,
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_f2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Best CBM test acc: {best_acc:.2f}%")
    if f1_results:
        print(f"Our zero-shot: {f1_results['our_zeroshot_acc']:.2f}% | "
              f"Paper linear-probe ceiling: {f1_results['paper_linear_probe_acc_336px']}%")


if __name__ == "__main__":
    main()
