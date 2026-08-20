"""Component 1b's follow-ups #1 and #2 (results/component1b_l1_vs_l2_ablation.md
§12), combined into one script since they share the same L1-SGD baseline and
Office-Home session:

#1 - ddo_lambda sweep: the analytic classifier's ddo_lambda was set to 1.0
    only to match the original alpha=1.0, never tuned. Sweeps it down to see
    whether a smaller value recovers most of the 1.75-point accuracy gap
    found at ddo_lambda=1.0, while keeping orthogonality meaningfully better
    than the L1-SGD baseline (mean_abs=0.4686, mean_sq=0.3903 - see
    component1b_l1_vs_l2_ablation.json).

#2 - Per-class breakdown at ddo_lambda=1.0 (the original setting): tests the
    mechanism hypothesis from component1b's own report §10 - that the
    accuracy gap concentrates in classes with less training data, rather
    than spreading evenly - by computing real per-class accuracy for both
    models and per-class training-sample counts, and checking whether the
    (L1_acc - L2_acc) gap actually correlates with sample count.
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
from data import get_office_home_datasets, OFFICE_HOME_DOMAINS
from experiments.domain_il import DomainILSession, JOINT_EPOCHS, RESULTS_DIR
from experiments.component1_l1_vs_l2_ablation import train_l1_sgd_joint, orthogonality_readout
from experiments.component1_analytic_domain_il import _build_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DDO_LAMBDA_SWEEP = [0.01, 0.03, 0.1, 0.3, 1.0]


def fit_analytic(session, model_ref, ddo_lambda, ridge_lambda=1.0):
    _, clf = _build_classifier(session, ridge_lambda=ridge_lambda, ddo_lambda=ddo_lambda)
    with torch.no_grad():
        for domain in session.domains:
            feats, labels, _ = session.caches[(domain, "train")]
            concept_acts = feats.to(session.device) @ model_ref.concept_embeddings.T
            clf.partial_fit(concept_acts, labels)
    return clf


def eval_per_domain(session, model_ref, clf):
    acc = {}
    with torch.no_grad():
        for d in session.domains:
            feats, labels, _ = session.caches[(d, "test")]
            feats, labels = feats.to(session.device), labels.to(session.device)
            concept_acts = feats @ model_ref.concept_embeddings.T
            logits = clf.predict_logits(concept_acts)
            acc[d] = 100.0 * (logits.argmax(dim=1) == labels).float().mean().item()
    return acc


def orthogonality_for_analytic(model_ref, clf):
    with torch.no_grad():
        diffs_flat = model_ref.diffs.reshape(-1, model_ref.diffs.shape[-1])
        concept_space = diffs_flat @ model_ref.concept_embeddings.T
        logits = clf.predict_logits(concept_space)
    return orthogonality_readout(logits)


def per_class_accuracy(session, predict_fn, num_classes, split="test"):
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for d in session.domains:
            feats, labels, _ = session.caches[(d, split)]
            feats, labels = feats.to(session.device), labels.to(session.device)
            preds = predict_fn(feats).argmax(dim=1)
            for c in range(num_classes):
                mask = labels == c
                n = mask.sum().item()
                if n:
                    total[c] += n
                    correct[c] += (preds[mask] == c).sum().item()
    return correct, total


def per_class_train_counts(session, num_classes):
    counts = torch.zeros(num_classes)
    for d in session.domains:
        _, labels, _ = session.caches[(d, "train")]
        for c in range(num_classes):
            counts[c] += (labels == c).sum().item()
    return counts


def main():
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "OfficeHome", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    session = DomainILSession(args, domains=OFFICE_HOME_DOMAINS, datasets_fn=get_office_home_datasets, dataset_key="OfficeHome")
    id2class = {v: k for k, v in session.classname2id.items()}
    num_classes = len(session.classname2id)

    logger.info("Training L1-SGD joint model once (shared across both follow-ups)...")
    l1_model, l1_acc = train_l1_sgd_joint(session, epochs=JOINT_EPOCHS)
    l1_ACC = sum(l1_acc.values()) / len(l1_acc)
    logger.info(f"L1-SGD: ACC={l1_ACC:.2f}")

    # ---- Follow-up #1: ddo_lambda sweep ----
    logger.info("Follow-up #1: ddo_lambda sweep on the analytic classifier...")
    sweep_results = []
    for ddo_lambda in DDO_LAMBDA_SWEEP:
        clf = fit_analytic(session, l1_model, ddo_lambda=ddo_lambda)
        acc = eval_per_domain(session, l1_model, clf)
        ACC = sum(acc.values()) / len(acc)
        orth = orthogonality_for_analytic(l1_model, clf)
        sweep_results.append({"ddo_lambda": ddo_lambda, "per_domain_test_acc": acc, "ACC": ACC, "orthogonality": orth})
        logger.info(f"  ddo_lambda={ddo_lambda}: ACC={ACC:.2f} (L1 ACC={l1_ACC:.2f}, gap={ACC - l1_ACC:+.2f}) "
                    f"orth_mean_abs={orth['mean_abs']:.4f} (L1={0.4686})")

    # ---- Follow-up #2: per-class breakdown at ddo_lambda=1.0 ----
    logger.info("Follow-up #2: per-class accuracy breakdown at ddo_lambda=1.0...")
    clf_10 = fit_analytic(session, l1_model, ddo_lambda=1.0)

    l1_correct, l1_total = per_class_accuracy(
        session, lambda feats: l1_model.forward_cached(feats)[1], num_classes)
    l2_correct, l2_total = per_class_accuracy(
        session, lambda feats: clf_10.predict_logits(feats @ l1_model.concept_embeddings.T), num_classes)
    train_counts = per_class_train_counts(session, num_classes)

    per_class = []
    for c in range(num_classes):
        l1_acc_c = 100.0 * l1_correct[c].item() / l1_total[c].item() if l1_total[c] > 0 else None
        l2_acc_c = 100.0 * l2_correct[c].item() / l2_total[c].item() if l2_total[c] > 0 else None
        gap = (l1_acc_c - l2_acc_c) if (l1_acc_c is not None and l2_acc_c is not None) else None
        per_class.append({
            "class": id2class[c], "train_samples": int(train_counts[c].item()),
            "l1_test_acc": l1_acc_c, "l2_test_acc": l2_acc_c,
            "gap_l1_minus_l2": gap,
        })

    valid = [p for p in per_class if p["gap_l1_minus_l2"] is not None]
    train_samples = np.array([p["train_samples"] for p in valid])
    gaps = np.array([p["gap_l1_minus_l2"] for p in valid])
    correlation = float(np.corrcoef(train_samples, gaps)[0, 1]) if len(valid) > 2 else None
    logger.info(f"Correlation(train_samples, L1-L2 accuracy gap) across {len(valid)} classes: {correlation}")

    results = {
        "component": "1b follow-ups - ddo_lambda sweep + per-class breakdown, Office-Home",
        "l1_sgd_baseline": {"per_domain_test_acc": l1_acc, "ACC": l1_ACC,
                              "orthogonality": {"mean_abs": 0.4686061143875122, "mean_sq": 0.39026784896850586}},
        "followup_1_ddo_lambda_sweep": sweep_results,
        "followup_2_per_class": {
            "per_class": per_class,
            "correlation_train_samples_vs_gap": correlation,
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component1b_officehome_followups.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
