"""Phase D remediation sweep - the same three fixes from experiments/
remediation.py (cumulative DDO, cached-embedding replay, EWC), applied to
Office-Home instead of PACS. Reuses CumulativeDDOSession/ReplaySession/
EWCSession unchanged (they were already dataset-agnostic - see
remediation.py) - only the dataset/domains/orderings differ.
"""
import os
import sys
import json
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from data import get_office_home_datasets, OFFICE_HOME_DOMAINS
from domain_il import STAGE_EPOCHS, compute_acc_bwt
from domain_il_officehome import DEFAULT_ORDERINGS
from remediation import (
    CumulativeDDOSession, ReplaySession, EWCSession,
    REPLAY_BUFFER_SIZE_PER_DOMAIN, EWC_LAMBDA, EWC_FISHER_SAMPLES, parse_local_args,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)

REMEDIATIONS = {
    "cumulative_ddo": CumulativeDDOSession,
    "replay": ReplaySession,
    "ewc": EWCSession,
}


def main():
    local_args = parse_local_args()
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "OfficeHome", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.class_avg_concept = True

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    stage_epochs = 2 if local_args.smoke_test else STAGE_EPOCHS
    orderings = (
        {"art_clipart_product_realworld": DEFAULT_ORDERINGS["art_clipart_product_realworld"]}
        if local_args.smoke_test else DEFAULT_ORDERINGS
    )
    output_name = "phase_d_remediation_smoke_results.json" if local_args.smoke_test else "phase_d_remediation_results.json"

    results = {
        "dataset": "OfficeHome",
        "domains": list(OFFICE_HOME_DOMAINS),
        "config": {
            "alpha": args.alpha, "beta": args.beta, "stage_epochs": stage_epochs,
            "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed, "replay_buffer_size_per_domain": REPLAY_BUFFER_SIZE_PER_DOMAIN,
            "ewc_lambda": EWC_LAMBDA, "ewc_fisher_samples": EWC_FISHER_SAMPLES,
        },
        "remediations": {},
    }

    session_kwargs = dict(domains=OFFICE_HOME_DOMAINS, datasets_fn=get_office_home_datasets, dataset_key="OfficeHome")

    for remediation_name, session_cls in REMEDIATIONS.items():
        print(f"\n=== Remediation: {remediation_name} ===")
        session = session_cls(args, **session_kwargs)
        runs = []
        for order_id, order in orderings.items():
            print(f"  order: {order_id}")
            stages, pre_metric = session.run_sequential(order, stage_epochs, checkpoint_stage0=False)
            ACC, BWT, R_diag, R_final = compute_acc_bwt(stages, order)
            runs.append({
                "order_id": order_id, "domain_order": order,
                "pre_training_ddo_erosion": pre_metric,
                "stages": stages, "R_diagonal": R_diag, "R_final": R_final,
                "ACC_final": ACC, "BWT": BWT,
            })
            print(f"    ACC={ACC:.2f} BWT={BWT:.2f}")
        results["remediations"][remediation_name] = runs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
