"""Component 4 - domain memory that never stores raw images (or any other
per-sample, per-record data), validated on PACS.

ReplaySession (experiments/remediation.py) keeps REPLAY_BUFFER_SIZE_PER_DOMAIN
(100) individually-sampled real cached CLIP embeddings per prior domain, and
replays them verbatim in every later training stage. Each stored 768-dim
vector still traces back 1:1 to one specific real training example - for a
domain like medical imaging, that's still a linkable per-patient record, just
in embedding form instead of pixels. GaussianReplaySession replaces that
per-sample buffer with a per-domain, per-class Gaussian summary (mean +
diagonal variance), computed once, then samples SYNTHETIC feature vectors
from that summary at every later stage - no individual real example (nor its
embedding) needs to survive past the moment its domain's summary is computed.

Not a novel algorithm - class-conditional Gaussian feature-space replay
without real exemplars is established in exemplar-free class-incremental
learning (closest published match: PASS, CVPR 2021's prototype + Gaussian
augmentation). Applied here to the domain-incremental forgetting problem this
project studies, with the privacy/data-retention motivation made explicit.

Two flagged deliberate simplifications (see planning/06 and the report for
why): (1) diagonal, not full, covariance - a full 768x768 covariance is both
expensive and severely under-determined from the handful-to-low-hundreds of
samples typically available per class per domain; (2) this class's synthetic
buffer is split EVENLY across a domain's classes, whereas ReplaySession
samples uniformly across a domain's pooled (possibly class-imbalanced)
examples - both are legitimate design choices, not a bug, but they mean the
two buffers aren't class-distribution-identical.

Caveat also worth stating plainly: this validation harness (like every other
session in domain_il.py/remediation.py) keeps every domain's full cached
embeddings in memory for the whole run, for convenience across orderings and
remediations sharing one process. That's a property of the shared simulation
harness, not of this mechanism - the per-class Gaussian summaries this class
computes and samples from are the ONLY thing a real, streaming deployment of
this remediation would need to retain past a domain's own training stage,
which is the property Component 4 actually claims and this comparison tests.
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from domain_il import DomainILSession, DEFAULT_ORDERINGS, STAGE_EPOCHS, PACS_DOMAINS, compute_acc_bwt
from remediation import ReplaySession, REPLAY_BUFFER_SIZE_PER_DOMAIN, parse_local_args

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)

MIN_STD = 1e-3


class GaussianReplaySession(DomainILSession):
    def __init__(self, args, buffer_size_per_domain=REPLAY_BUFFER_SIZE_PER_DOMAIN, min_std=MIN_STD, **session_kwargs):
        super().__init__(args, **session_kwargs)
        self.buffer_size_per_domain = buffer_size_per_domain
        self.min_std = min_std
        self._domain_class_gaussians = {}  # domain -> {class_id: (mean, std)}

    def _summarize_domain(self, domain):
        if domain not in self._domain_class_gaussians:
            feats, labels, _attrs = self.caches[(domain, "train")]
            stats = {}
            for c in labels.unique().tolist():
                class_feats = feats[labels == c]
                mean = class_feats.mean(dim=0)
                std = class_feats.std(dim=0, unbiased=False).clamp_min(self.min_std)
                stats[c] = (mean, std)
            self._domain_class_gaussians[domain] = stats
        return self._domain_class_gaussians[domain]

    def _build_stage_loader(self, domain, stage_idx, domain_order):
        base_ds = TensorDataset(*self.caches[(domain, "train")])
        prior_domains = domain_order[:stage_idx]
        if not prior_domains:
            return DataLoader(base_ds, batch_size=self.args.batch_size, shuffle=True)

        num_concepts = self.caches[(domain, "train")][2].shape[1]
        rng = torch.Generator().manual_seed(self.args.seed + stage_idx)
        replay_parts = []
        for d in prior_domains:
            stats = self._summarize_domain(d)
            classes = sorted(stats.keys())
            per_class = max(1, self.buffer_size_per_domain // len(classes))
            syn_feats, syn_labels = [], []
            for c in classes:
                mean, std = stats[c]
                noise = torch.randn(per_class, mean.shape[0], generator=rng)
                syn_feats.append(mean.unsqueeze(0) + noise * std.unsqueeze(0))
                syn_labels.append(torch.full((per_class,), c, dtype=torch.long))
            syn_feats = torch.cat(syn_feats, dim=0)
            syn_labels = torch.cat(syn_labels, dim=0)
            syn_attrs = torch.zeros(syn_feats.size(0), num_concepts)
            replay_parts.append(TensorDataset(syn_feats, syn_labels, syn_attrs))
        combined = ConcatDataset([base_ds] + replay_parts)
        return DataLoader(combined, batch_size=self.args.batch_size, shuffle=True)


CONDITIONS = {
    "naive": DomainILSession,
    "replay_real": ReplaySession,
    "replay_gaussian": GaussianReplaySession,
}


def run_conditions(args, orderings, session_kwargs=None, stage_epochs=STAGE_EPOCHS):
    session_kwargs = session_kwargs or {}
    results_by_condition = {}
    for name, session_cls in CONDITIONS.items():
        print(f"\n=== Condition: {name} ===")
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
        results_by_condition[name] = runs
    return results_by_condition


def main():
    local_args = parse_local_args()
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "PACS", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.class_avg_concept = True

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    stage_epochs = 2 if local_args.smoke_test else STAGE_EPOCHS
    orderings = (
        {"photo_art_cartoon_sketch": DEFAULT_ORDERINGS["photo_art_cartoon_sketch"]}
        if local_args.smoke_test else DEFAULT_ORDERINGS
    )
    output_name = "component4_pacs_smoke_results.json" if local_args.smoke_test else "component4_pacs_results.json"

    results_by_condition = run_conditions(args, orderings, stage_epochs=stage_epochs)

    results = {
        "dataset": "PACS",
        "domains": list(PACS_DOMAINS),
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
