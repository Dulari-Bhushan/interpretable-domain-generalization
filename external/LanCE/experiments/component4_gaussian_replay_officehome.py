"""Component 4, Office-Home - the same naive/real-replay/Gaussian-replay
3-way comparison from component4_gaussian_replay.py (PACS), applied to
Office-Home instead. Reuses GaussianReplaySession/CONDITIONS unchanged
(dataset-agnostic, same as remediation.py's sessions) - only the
dataset/domains/orderings differ, mirroring remediation_officehome.py's
own import-from-remediation.py pattern.
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
from domain_il import STAGE_EPOCHS
from domain_il_officehome import DEFAULT_ORDERINGS
from remediation import REPLAY_BUFFER_SIZE_PER_DOMAIN, parse_local_args
from component4_gaussian_replay import CONDITIONS, MIN_STD, run_conditions

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)


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
    output_name = "component4_officehome_smoke_results.json" if local_args.smoke_test else "component4_officehome_results.json"

    session_kwargs = dict(domains=OFFICE_HOME_DOMAINS, datasets_fn=get_office_home_datasets, dataset_key="OfficeHome")
    results_by_condition = run_conditions(args, orderings, session_kwargs=session_kwargs, stage_epochs=stage_epochs)

    results = {
        "dataset": "OfficeHome",
        "domains": list(OFFICE_HOME_DOMAINS),
        "config": {
            "alpha": args.alpha, "beta": args.beta, "stage_epochs": stage_epochs,
            "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed, "replay_buffer_size_per_domain": REPLAY_BUFFER_SIZE_PER_DOMAIN,
            "min_std": MIN_STD,
        },
        "conditions": results_by_condition,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
