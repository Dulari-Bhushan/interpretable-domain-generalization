"""Phase D - repeat Phase B's Domain-IL sequential protocol on Office-Home
(4 domains, 65 classes) instead of PACS (4 domains, 7 classes).

Why: Phase B found real forgetting in only 1 of 3 PACS domain orderings,
and the joint/oracle model already hit 98.3% ACC - PACS's near-ceiling
accuracy on 7 visually-distinct classes left little room to observe
forgetting even if the underlying mechanism is fragile. Office-Home's 65
classes and far fewer images per class per domain (~30-70/class vs PACS's
~200-450/class) should leave much less headroom, giving forgetting more
room to show up if it's really there.

Reuses experiments/domain_il.py's DomainILSession unchanged (it was
generalized specifically for this - domains/datasets_fn/dataset_key are
now constructor parameters instead of hardcoded PACS values) and
data/__init__.py's get_office_home_datasets (mirrors get_pacs_datasets).
Same protocol, metrics (ACC/BWT/DDO-erosion), and hyperparameters as
Phase B, deliberately unchanged so results are comparable, not confounded
by an unrelated hyperparameter difference.
"""
import os
import sys
import json
import random
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from data import get_office_home_datasets, OFFICE_HOME_DOMAINS
from domain_il import DomainILSession, compute_acc_bwt, parse_local_args, STAGE_EPOCHS, JOINT_EPOCHS

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)

DEFAULT_ORDERINGS = {
    "art_clipart_product_realworld": ["art", "clipart", "product", "real_world"],
    "realworld_product_clipart_art": ["real_world", "product", "clipart", "art"],
    "clipart_realworld_art_product": ["clipart", "real_world", "art", "product"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("domain_il_officehome.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    local_args = parse_local_args()
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "OfficeHome", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    stage_epochs = 2 if local_args.smoke_test else local_args.stage_epochs
    joint_epochs = 2 if local_args.smoke_test else local_args.joint_epochs
    orderings = (
        {"art_clipart_product_realworld": DEFAULT_ORDERINGS["art_clipart_product_realworld"]}
        if local_args.smoke_test else DEFAULT_ORDERINGS
    )
    output_name = "phase_d_smoke_results.json" if local_args.smoke_test else "phase_d_results.json"

    session = DomainILSession(
        args, domains=OFFICE_HOME_DOMAINS, datasets_fn=get_office_home_datasets, dataset_key="OfficeHome"
    )

    logger.info("Running joint/oracle...")
    joint_acc = session.run_joint(joint_epochs)
    logger.info(f"Joint/oracle per-domain acc: {joint_acc}")

    results = {
        "dataset": "OfficeHome",
        "domains": list(OFFICE_HOME_DOMAINS),
        "classes": list(session.classname2id.keys()),
        "config": {
            "alpha": args.alpha, "beta": args.beta, "CBM_type": args.CBM_type,
            "CLIP_type": args.CLIP_type, "stage_epochs": stage_epochs, "joint_epochs": joint_epochs,
            "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        },
        "joint_oracle": {"per_domain_test_acc": joint_acc, "ACC": sum(joint_acc.values()) / len(joint_acc)},
        "sequential_runs": [],
    }

    first_order_id = next(iter(orderings))
    for order_id, order in orderings.items():
        logger.info(f"Running sequential order: {order_id} = {order}")
        stages, pre_metric = session.run_sequential(
            order, stage_epochs, checkpoint_stage0=(order_id == first_order_id and not local_args.smoke_test)
        )
        ACC, BWT, R_diag, R_final = compute_acc_bwt(stages, order)
        results["sequential_runs"].append({
            "order_id": order_id, "domain_order": order,
            "pre_training_ddo_erosion": pre_metric,
            "stages": stages, "R_diagonal": R_diag, "R_final": R_final,
            "ACC_final": ACC, "BWT": BWT,
        })
        logger.info(f"{order_id}: ACC={ACC:.2f} BWT={BWT:.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
