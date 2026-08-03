"""Phase C - remediation attempts: does a textbook CL fix already solve it?

Same PACS Domain-IL harness as Phase B (experiments/domain_il.py), three
standard-toolkit fixes layered on top of the naive-sequential protocol via
DomainILSession's Phase-C-ready seams (between_stage_hooks, _compute_loss,
_build_stage_loader) - run_sequential's control flow is untouched.

1. Cumulative DDO (near-free).
   Note on adapting the plan's original framing: LanCE's DDO regularizer
   (Eq. 12/13) is computed from a fixed, dataset-agnostic ~204-descriptor
   pool that never changes per training domain - it's already applied
   identically at every stage of Phase B's naive-sequential baseline, so
   there's no *per-domain* DDO term to "accumulate" in the literal sense
   the plan describes for LADA/CUB's single source->target setup. For
   PACS's actual 4-domain sequence, this implements the natural
   adaptation instead: at each stage, additionally regularize the
   classifier toward orthogonality with the mean cached-image-feature
   direction of every domain trained on so far (not the current one) -
   still purely feature-based, no raw images, still "near-free" in the
   sense the plan means (reuses caches already on disk).

2. Cached-embedding replay.
   Store a subsample (REPLAY_BUFFER_SIZE_PER_DOMAIN) of each prior
   domain's cached train features, mix into every subsequent stage's
   training batches. A real memory cost, not free.

3. EWC-style regularization on W_F (Kirkpatrick et al.).
   After each stage, compute a diagonal empirical-Fisher approximation
   over the classifier's parameters from that domain's own training data,
   and penalize drift from the post-stage parameter values in every
   subsequent stage, weighted by that Fisher importance. Standard
   multi-task EWC: penalties from every prior stage are summed.
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

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)

REPLAY_BUFFER_SIZE_PER_DOMAIN = 100
EWC_LAMBDA = 1000.0
EWC_FISHER_SAMPLES = 200  # empirical-Fisher estimate over a subsample of the stage's own train data


class CumulativeDDOSession(DomainILSession):
    def __init__(self, args, **session_kwargs):
        super().__init__(args, **session_kwargs)
        self._seen_domains = []
        self.between_stage_hooks.append(self._record_seen_domain)

    def _record_seen_domain(self, session, stage_idx, domain, model, optimizer):
        self._seen_domains.append(domain)

    def run_sequential(self, domain_order, epochs_per_stage=STAGE_EPOCHS, checkpoint_stage0=False):
        self._seen_domains = []
        return super().run_sequential(domain_order, epochs_per_stage, checkpoint_stage0)

    def _compute_loss(self, cls_preds, labels, reg, model=None, extra=None):
        base_loss = super()._compute_loss(cls_preds, labels, reg, model=model, extra=extra)
        if not self._seen_domains or model is None:
            return base_loss
        centroids = torch.stack(
            [self.caches[(d, "train")][0].mean(dim=0) for d in self._seen_domains]
        ).to(self.device)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True)
        extra_reg = model.classifier[1:](centroids @ model.concept_embeddings.T)
        extra_loss = torch.abs(extra_reg).mean()
        return base_loss + self.args.alpha * extra_loss


class ReplaySession(DomainILSession):
    def __init__(self, args, buffer_size_per_domain=REPLAY_BUFFER_SIZE_PER_DOMAIN, **session_kwargs):
        super().__init__(args, **session_kwargs)
        self.buffer_size_per_domain = buffer_size_per_domain

    def _build_stage_loader(self, domain, stage_idx, domain_order):
        base_ds = TensorDataset(*self.caches[(domain, "train")])
        prior_domains = domain_order[:stage_idx]
        if not prior_domains:
            return DataLoader(base_ds, batch_size=self.args.batch_size, shuffle=True)

        rng = torch.Generator().manual_seed(self.args.seed + stage_idx)
        replay_parts = []
        for d in prior_domains:
            feats, labels, attrs = self.caches[(d, "train")]
            n = feats.size(0)
            k = min(self.buffer_size_per_domain, n)
            idx = torch.randperm(n, generator=rng)[:k]
            replay_parts.append(TensorDataset(feats[idx], labels[idx], attrs[idx]))
        combined = ConcatDataset([base_ds] + replay_parts)
        return DataLoader(combined, batch_size=self.args.batch_size, shuffle=True)


class EWCSession(DomainILSession):
    def __init__(self, args, ewc_lambda=EWC_LAMBDA, **session_kwargs):
        super().__init__(args, **session_kwargs)
        self.ewc_lambda = ewc_lambda
        self._ewc_terms = []  # list of (param_snapshot: dict, fisher: dict)
        self.between_stage_hooks.append(self._consolidate_ewc)

    def run_sequential(self, domain_order, epochs_per_stage=STAGE_EPOCHS, checkpoint_stage0=False):
        self._ewc_terms = []
        return super().run_sequential(domain_order, epochs_per_stage, checkpoint_stage0)

    def _consolidate_ewc(self, session, stage_idx, domain, model, optimizer):
        """Diagonal empirical Fisher information over classifier params,
        estimated from a subsample of the just-finished stage's own
        training data (squared gradient of the sampled-label log-likelihood,
        averaged)."""
        model.eval()
        params = {n: p for n, p in model.classifier.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}

        feats, labels, _attrs = self.caches[(domain, "train")]
        n = feats.size(0)
        rng = torch.Generator().manual_seed(self.args.seed + stage_idx)
        idx = torch.randperm(n, generator=rng)[: min(EWC_FISHER_SAMPLES, n)]

        for i in idx.tolist():
            feat = feats[i : i + 1].to(self.device)
            label = labels[i : i + 1].to(self.device)
            model.zero_grad(set_to_none=True)
            _, cls_preds, _ = model.forward_cached(feat)
            log_prob = torch.log_softmax(cls_preds, dim=1)[0, label.item()]
            log_prob.backward()
            for n_, p in params.items():
                if p.grad is not None:
                    fisher[n_] += p.grad.detach() ** 2

        n_samples = max(len(idx), 1)
        for n_ in fisher:
            fisher[n_] /= n_samples
        snapshot = {n_: p.detach().clone() for n_, p in params.items()}
        self._ewc_terms.append((snapshot, fisher))
        model.zero_grad(set_to_none=True)
        model.train()

    def _compute_loss(self, cls_preds, labels, reg, model=None, extra=None):
        base_loss = super()._compute_loss(cls_preds, labels, reg, model=model, extra=extra)
        if not self._ewc_terms or model is None:
            return base_loss
        params = {n: p for n, p in model.classifier.named_parameters() if p.requires_grad}
        penalty = torch.tensor(0.0, device=self.device)
        for snapshot, fisher in self._ewc_terms:
            for n, p in params.items():
                penalty = penalty + (fisher[n] * (p - snapshot[n]) ** 2).sum()
        return base_loss + self.ewc_lambda * penalty


REMEDIATIONS = {
    "cumulative_ddo": CumulativeDDOSession,
    "replay": ReplaySession,
    "ewc": EWCSession,
}


def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_test", action="store_true", default=False,
                         help="2 epochs/stage, 1 ordering, writes phase_c_smoke_results.json")
    local_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return local_args


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
    output_name = "phase_c_smoke_results.json" if local_args.smoke_test else "phase_c_results.json"

    results = {
        "dataset": "PACS",
        "domains": list(PACS_DOMAINS),
        "config": {
            "alpha": args.alpha, "beta": args.beta, "stage_epochs": stage_epochs,
            "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed, "replay_buffer_size_per_domain": REPLAY_BUFFER_SIZE_PER_DOMAIN,
            "ewc_lambda": EWC_LAMBDA, "ewc_fisher_samples": EWC_FISHER_SAMPLES,
        },
        "remediations": {},
    }

    for remediation_name, session_cls in REMEDIATIONS.items():
        print(f"\n=== Remediation: {remediation_name} ===")
        session = session_cls(args)
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
