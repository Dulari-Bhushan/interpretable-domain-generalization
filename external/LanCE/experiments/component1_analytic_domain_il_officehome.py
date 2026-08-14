"""Component 1 validation on Office-Home - the harder benchmark where Phase D
found forgetting in every ordering (BWT -0.68 to -4.68), not just 1 of 3 like
PACS. This is the real test: PACS's near-ceiling accuracy left little room
for the analytic classifier's exactness claim to look impressive (baseline
SGD was already close to the oracle in 2 of 3 orderings there). Office-Home
is where the original remediations (cumulative DDO, replay) only partially
closed the gap - if the analytic classifier's "max diff from joint = 0" result
holds here too, that's the real evidence, not the PACS smoke test.

Reuses every function from component1_analytic_domain_il.py unchanged - only
the dataset (domains/datasets_fn/dataset_key) and output path differ, exactly
mirroring how domain_il_officehome.py reuses domain_il.py's DomainILSession.
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
from experiments.domain_il import DomainILSession, compute_acc_bwt, RESULTS_DIR
from experiments.component1_analytic_domain_il import run_analytic_joint, run_analytic_sequential

DEFAULT_ORDERINGS = {
    "art_clipart_product_realworld": ["art", "clipart", "product", "real_world"],
    "realworld_product_clipart_art": ["real_world", "product", "clipart", "art"],
    "clipart_realworld_art_product": ["clipart", "real_world", "art", "product"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "OfficeHome", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ridge_lambda, ddo_lambda = 1.0, 1.0

    session = DomainILSession(
        args, domains=OFFICE_HOME_DOMAINS, datasets_fn=get_office_home_datasets, dataset_key="OfficeHome"
    )

    logger.info("Running analytic joint/oracle fit (all 4 Office-Home domains pooled)...")
    joint_acc = run_analytic_joint(session, ridge_lambda, ddo_lambda)
    logger.info(f"Joint/oracle per-domain acc: { {k: round(v,2) for k,v in joint_acc.items()} }")
    joint_ACC = sum(joint_acc.values()) / len(joint_acc)

    results = {
        "dataset": "OfficeHome",
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
    out_path = os.path.join(RESULTS_DIR, "component1_officehome_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
