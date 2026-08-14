"""Component 1 validation: does the analytic incremental classifier
(model/analytic_classifier.py) actually deliver on the "exact, zero
forgetting" claim, on the same PACS harness and same cached CLIP embeddings
Phase B/C already used?

Reuses DomainILSession from domain_il.py purely as a source of already-built
CLIP embedding caches, concept_embeddings, and domain_diffs - nothing here
does gradient descent. For each of Phase B's 3 domain orderings, this fits
the analytic classifier one domain at a time (no replay, no revisiting
earlier domains) and checks the result against a single joint fit over all
4 domains pooled at once. If the exactness claim is correct, BWT should
land at (numerically) zero and every ordering's final per-domain accuracy
should match the joint fit's, not just be close to it.
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
from experiments.domain_il import DomainILSession, compute_acc_bwt, DEFAULT_ORDERINGS, RESULTS_DIR
from model.analytic_classifier import AnalyticDomainIncrementalClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_classifier(session, ridge_lambda, ddo_lambda):
    model = session._build_model()  # only used for its fixed concept_embeddings + diffs
    clf = AnalyticDomainIncrementalClassifier(
        concept_dim=len(session.concept2id),
        num_classes=len(session.classname2id),
        domain_diffs=model.diffs,
        concept_embeddings=model.concept_embeddings,
        ridge_lambda=ridge_lambda,
        ddo_lambda=ddo_lambda,
        device=session.device,
    )
    return model, clf


def _eval_all_domains(session, model, clf):
    accs = {}
    with torch.no_grad():
        for d in session.domains:
            feats, labels, _ = session.caches[(d, "test")]
            feats, labels = feats.to(session.device), labels.to(session.device)
            concept_acts = feats @ model.concept_embeddings.T
            logits = clf.predict_logits(concept_acts)
            accs[d] = 100.0 * (logits.argmax(dim=1) == labels).float().mean().item()
    return accs


def run_analytic_joint(session, ridge_lambda, ddo_lambda):
    model, clf = _build_classifier(session, ridge_lambda, ddo_lambda)
    with torch.no_grad():
        for domain in session.domains:
            feats, labels, _ = session.caches[(domain, "train")]
            concept_acts = feats.to(session.device) @ model.concept_embeddings.T
            clf.partial_fit(concept_acts, labels)
    return _eval_all_domains(session, model, clf)


def run_analytic_sequential(session, domain_order, ridge_lambda, ddo_lambda):
    model, clf = _build_classifier(session, ridge_lambda, ddo_lambda)
    stages = []
    with torch.no_grad():
        for stage_idx, domain in enumerate(domain_order):
            feats, labels, _ = session.caches[(domain, "train")]
            concept_acts = feats.to(session.device) @ model.concept_embeddings.T
            clf.partial_fit(concept_acts, labels)

            per_domain_acc = _eval_all_domains(session, model, clf)
            stages.append({"stage": stage_idx, "trained_domain": domain, "per_domain_test_acc": per_domain_acc})
            logger.info(f"[stage {stage_idx}] trained={domain} "
                        f"acc={ {k: round(v, 2) for k, v in per_domain_acc.items()} }")
    return stages


def main():
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "PACS", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ridge_lambda, ddo_lambda = 1.0, 1.0

    session = DomainILSession(args)

    logger.info("Running analytic joint/oracle fit (all 4 domains pooled)...")
    joint_acc = run_analytic_joint(session, ridge_lambda, ddo_lambda)
    logger.info(f"Joint/oracle per-domain acc: { {k: round(v,2) for k,v in joint_acc.items()} }")
    joint_ACC = sum(joint_acc.values()) / len(joint_acc)

    results = {
        "dataset": "PACS",
        "component": "1 - analytic domain-incremental classifier",
        "config": {"ridge_lambda": ridge_lambda, "ddo_lambda": ddo_lambda, "seed": args.seed},
        "joint_oracle": {"per_domain_test_acc": joint_acc, "ACC": joint_ACC},
        "sequential_runs": [],
    }

    for order_id, order in DEFAULT_ORDERINGS.items():
        logger.info(f"Running analytic sequential order: {order_id} = {order}")
        stages = run_analytic_sequential(session, order, ridge_lambda, ddo_lambda)
        ACC, BWT, R_diag, R_final = compute_acc_bwt(stages, order)
        max_abs_diff_from_joint = max(abs(R_final[d] - joint_acc[d]) for d in R_final)
        results["sequential_runs"].append({
            "order_id": order_id, "domain_order": order,
            "stages": stages, "R_diagonal": R_diag, "R_final": R_final,
            "ACC_final": ACC, "BWT": BWT,
            "max_abs_diff_from_joint_acc": max_abs_diff_from_joint,
        })
        logger.info(f"{order_id}: ACC={ACC:.4f} BWT={BWT:.4f} "
                    f"max|diff from joint|={max_abs_diff_from_joint:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component1_pacs_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
