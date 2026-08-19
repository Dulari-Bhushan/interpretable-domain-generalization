"""Component 1, improvement #1: does the L1->L2 DDO substitution change anything
that matters, under otherwise-identical conditions?

Component 1's exact incremental classifier (model/analytic_classifier.py) had to
swap DDO's original L1 (mean-absolute) orthogonality penalty for an L2
(mean-square) surrogate to get a closed-form solution. This script runs both
versions - original SGD+L1-DDO (exactly Phase 0/B/D's own training loop,
unmodified) and the analytic L2-DDO classifier - on the same pooled joint data,
same domains, same descriptor pool, and reports two things for each:
  1. Final classification accuracy, per domain and overall.
  2. The orthogonality property DDO actually optimizes for: how close to zero
     the classifier's response to every (descriptor, anchor-class) direction is,
     reported both as mean(|.|) - the metric domain_il.py already tracks as
     "ddo_erosion" - and mean(.^2), so both models are read on both scales.

No new data pass, no new CLIP encoding - reuses the exact cached embeddings
Phase B/D already built.
"""
import os
import sys
import json
import random
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from data import get_pacs_datasets, PACS_DOMAINS, get_office_home_datasets, OFFICE_HOME_DOMAINS
from experiments.domain_il import DomainILSession, JOINT_EPOCHS, RESULTS_DIR
from experiments.component1_analytic_domain_il import run_analytic_joint, _build_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_l1_sgd_joint(session, epochs=JOINT_EPOCHS):
    """Exactly session.run_joint(), except it also hands back the trained
    model (run_joint discards it) so the orthogonality metric can be read
    off afterward."""
    from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
    model = session._build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=session.args.lr, weight_decay=session.args.weight_decay)
    pooled = ConcatDataset([TensorDataset(*session.caches[(d, "train")]) for d in session.domains])
    loader = DataLoader(pooled, batch_size=session.args.batch_size, shuffle=True)
    session._train_loop(model, optimizer, loader, epochs)
    per_domain_acc = session._eval_all_domains(model)
    return model, per_domain_acc


def orthogonality_readout(diffs_flat_logits):
    """Both scales, on the same tensor, so L1- and L2-trained models are
    compared on identical footing regardless of which one they were
    optimized for."""
    return {
        "mean_abs": torch.abs(diffs_flat_logits).mean().item(),
        "mean_sq": (diffs_flat_logits ** 2).mean().item(),
    }


def run_for_dataset(dataset_key, domains, datasets_fn):
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = dataset_key, "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    session = DomainILSession(args, domains=domains, datasets_fn=datasets_fn, dataset_key=dataset_key)

    logger.info(f"[{dataset_key}] Training original SGD + L1-DDO joint model ({JOINT_EPOCHS} epochs)...")
    l1_model, l1_acc = train_l1_sgd_joint(session)
    l1_ACC = sum(l1_acc.values()) / len(l1_acc)
    with torch.no_grad():
        l1_diffs_flat = l1_model.diffs.reshape(-1, l1_model.diffs.shape[-1])
        l1_logits = l1_model.classifier[1:](l1_diffs_flat @ l1_model.concept_embeddings.T)
    l1_orth = orthogonality_readout(l1_logits)
    logger.info(f"[{dataset_key}] L1-SGD: ACC={l1_ACC:.2f} per-domain={ {k: round(v,2) for k,v in l1_acc.items()} } "
                f"orth={ {k: round(v,4) for k,v in l1_orth.items()} }")

    logger.info(f"[{dataset_key}] Fitting analytic L2-DDO joint model...")
    l2_model_ref, l2_clf = _build_classifier(session, ridge_lambda=1.0, ddo_lambda=1.0)
    with torch.no_grad():
        for domain in session.domains:
            feats, labels, _ = session.caches[(domain, "train")]
            concept_acts = feats.to(session.device) @ l2_model_ref.concept_embeddings.T
            l2_clf.partial_fit(concept_acts, labels)
    l2_acc = {}
    with torch.no_grad():
        for d in session.domains:
            feats, labels, _ = session.caches[(d, "test")]
            feats, labels = feats.to(session.device), labels.to(session.device)
            concept_acts = feats @ l2_model_ref.concept_embeddings.T
            logits = l2_clf.predict_logits(concept_acts)
            l2_acc[d] = 100.0 * (logits.argmax(dim=1) == labels).float().mean().item()
    l2_ACC = sum(l2_acc.values()) / len(l2_acc)
    with torch.no_grad():
        l2_diffs_flat = l2_model_ref.diffs.reshape(-1, l2_model_ref.diffs.shape[-1])
        l2_concept_space = l2_diffs_flat @ l2_model_ref.concept_embeddings.T
        l2_logits = l2_clf.predict_logits(l2_concept_space)
    l2_orth = orthogonality_readout(l2_logits)
    logger.info(f"[{dataset_key}] L2-analytic: ACC={l2_ACC:.2f} per-domain={ {k: round(v,2) for k,v in l2_acc.items()} } "
                f"orth={ {k: round(v,4) for k,v in l2_orth.items()} }")

    return {
        "dataset": dataset_key,
        "l1_sgd": {"per_domain_test_acc": l1_acc, "ACC": l1_ACC, "orthogonality": l1_orth},
        "l2_analytic": {"per_domain_test_acc": l2_acc, "ACC": l2_ACC, "orthogonality": l2_orth},
        "ACC_diff_l2_minus_l1": l2_ACC - l1_ACC,
    }


def main():
    results = {
        "component": "1b - L1 (original SGD) vs L2 (analytic) DDO ablation",
        "config": {"joint_epochs": JOINT_EPOCHS, "batch_size": 64, "lr": 1e-4, "weight_decay": 1e-4,
                    "ridge_lambda": 1.0, "ddo_lambda": 1.0, "alpha": 1.0, "seed": 0},
        "datasets": [],
    }
    results["datasets"].append(run_for_dataset("PACS", PACS_DOMAINS, get_pacs_datasets))
    results["datasets"].append(run_for_dataset("OfficeHome", OFFICE_HOME_DOMAINS, get_office_home_datasets))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component1b_l1_vs_l2_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
