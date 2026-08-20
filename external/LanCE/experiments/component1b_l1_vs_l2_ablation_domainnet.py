"""Component 1b's follow-up #3 (results/component1b_l1_vs_l2_ablation.md
§12): carry the L1-vs-L2 DDO ablation forward to DomainNet, since the
original finding (free on PACS, costly on Office-Home) was dataset-
dependent, not universal - this checks whether that pattern continues,
reverses, or does something new at DomainNet's scale (345 classes, 6
domains, ~586K images).

Reuses run_for_dataset from component1_l1_vs_l2_ablation.py unchanged -
only the dataset differs. Written as a separate script (not added to that
module's main()) so PACS/Office-Home don't get needlessly retrained every
time this runs.
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_domainnet_datasets, DOMAINNET_DOMAINS
from experiments.domain_il import RESULTS_DIR
from experiments.component1_l1_vs_l2_ablation import run_for_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    result = run_for_dataset("DomainNet", DOMAINNET_DOMAINS, get_domainnet_datasets)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "component1b_l1_vs_l2_ablation_domainnet.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
