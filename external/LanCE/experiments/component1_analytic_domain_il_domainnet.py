"""Component 1's scale test: does the exact-match result (results/component1_
exact_classifier.md) still hold at DomainNet's scale - 6 domains, 345 classes,
~0.6M images - not just PACS/Office-Home's 4 domains, 7-65 classes?

Reuses every function from component1_analytic_domain_il.py unchanged - only
the dataset (domains/datasets_fn/dataset_key) and output path differ, exactly
mirroring how the Office-Home version reuses the PACS version's code.

Note on domain orderings: DomainNet has 6 domains, so a "3 orderings" sweep
isn't directly comparable to PACS/Office-Home's exhaustive-ish coverage of a
4-domain space (3 of 24 possible vs. 3 of 720 possible here) - kept at 3 for
consistency with every other phase's reporting convention, not because it's
equally representative. Flagged honestly rather than silently reused.
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
from data import get_domainnet_datasets, DOMAINNET_DOMAINS
from experiments.domain_il import DomainILSession, compute_acc_bwt, RESULTS_DIR
from experiments.component1_analytic_domain_il import run_analytic_joint, run_analytic_sequential

DEFAULT_ORDERINGS = {
    "real_clipart_painting_sketch_infograph_quickdraw":
        ["real", "clipart", "painting", "sketch", "infograph", "quickdraw"],
    "quickdraw_infograph_sketch_painting_clipart_real":
        ["quickdraw", "infograph", "sketch", "painting", "clipart", "real"],
    "clipart_real_quickdraw_infograph_painting_sketch":
        ["clipart", "real", "quickdraw", "infograph", "painting", "sketch"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "DomainNet", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ridge_lambda, ddo_lambda = 1.0, 1.0

    session = DomainILSession(
        args, domains=DOMAINNET_DOMAINS, datasets_fn=get_domainnet_datasets, dataset_key="DomainNet"
    )

    logger.info("Running analytic joint/oracle fit (all 6 DomainNet domains pooled)...")
    joint_acc = run_analytic_joint(session, ridge_lambda, ddo_lambda)
    logger.info(f"Joint/oracle per-domain acc: { {k: round(v,2) for k,v in joint_acc.items()} }")
    joint_ACC = sum(joint_acc.values()) / len(joint_acc)

    results = {
        "dataset": "DomainNet",
        "component": "1 - analytic domain-incremental classifier (scale test)",
        "num_classes": 345,
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
    out_path = os.path.join(RESULTS_DIR, "component1_domainnet_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
