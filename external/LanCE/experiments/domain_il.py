"""Phase B - Domain-IL sequential protocol on PACS (the core forgetting probe).

LanCE's DDO regularizer is optimized once, jointly, on a single training
domain (Sec. 4.1) - there is no update rule for domains arriving later.
This script simulates the continual-learning setting LanCE was never
evaluated in: train on PACS domain i with CE+DDO, evaluate on all 4
domains, move to domain i+1 fine-tuning the same model (no replay), repeat.

Two conditions:
  (a) Joint/oracle - train once on all 4 domains pooled (non-continual
      upper bound).
  (b) Naive sequential - the actual Domain-IL protocol, repeated across
      multiple domain orderings (order-sensitivity is a known CL confound).

Reported in ACC/BWT (Lopez-Paz & Ranzato), plus a mechanism-level metric
at every stage boundary: the DDO orthogonality loss recomputed against the
model's own domain_diffs (fixed since construction) using the *current*
classifier weights - `|classifier[1:](diffs @ concept_embeddings.T)|.mean()`.
This needs no data pass (diffs/concept_embeddings don't depend on any
image), so it's a free per-stage measurement of whether the orthogonality
property DDO trained for is silently eroding even if raw accuracy holds up.

All training reuses the cached-embedding pipeline from train_cached.py/
cache_utils.py (CLIP is frozen, so each PACS domain's image features are
computed once and reused across every ordering + the joint run).
"""
import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from model.cbm_models import clip_cbm_orth
from data import get_pacs_datasets, PACS_DOMAINS
from cache_utils import get_or_build_feature_cache

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)

DEFAULT_ORDERINGS = {
    "photo_art_cartoon_sketch": ["photo", "art_painting", "cartoon", "sketch"],
    "sketch_cartoon_art_photo": ["sketch", "cartoon", "art_painting", "photo"],
    "art_sketch_photo_cartoon": ["art_painting", "sketch", "photo", "cartoon"],
}
STAGE_EPOCHS = 50
JOINT_EPOCHS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("domain_il.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DomainILSession:
    """Builds a dataset's domains + their CLIP-embedding caches once; reused
    across the joint run and every sequential ordering.

    Defaults to PACS (Phase B/C's original dataset) for full backward
    compatibility. Phase D reuses this same class for Office-Home by passing
    domains/datasets_fn/dataset_key explicitly - e.g.:
        DomainILSession(args, domains=OFFICE_HOME_DOMAINS,
                         datasets_fn=get_office_home_datasets, dataset_key="OfficeHome")
    Nothing else in this file (or in experiments/remediation.py, which
    subclasses this) needs to change per dataset.
    """

    def __init__(self, args, domains=None, datasets_fn=None, dataset_key=None):
        self.args = args
        self.device = args.device
        self.domains = domains if domains is not None else PACS_DOMAINS
        self.dataset_key = dataset_key if dataset_key is not None else "PACS"
        build_datasets = datasets_fn if datasets_fn is not None else get_pacs_datasets
        self.datasets, self.classname2id, self.concept2id, self.domain_diffs = build_datasets(args)
        self._build_caches()
        # Phase C readiness: empty in Phase B. Each hook is called as
        # hook(session, stage_idx, domain, model, optimizer) right after a
        # stage's results are logged - this is where cumulative-DDO/replay/
        # EWC would plug in later without touching run_sequential itself.
        self.between_stage_hooks = []

    def _build_caches(self):
        clip_model, _ = clip.load(self.args.CLIP_type, device=self.device)
        for p in clip_model.parameters():
            p.requires_grad = False
        cache_prefix = f"{self.dataset_key}_{self.args.CLIP_type}".replace("/", "-")
        self.caches = {}
        for domain in self.domains:
            for split in ("train", "test"):
                name = f"{cache_prefix}_{domain}_{split}"
                feats, labels, attrs = get_or_build_feature_cache(
                    name, self.datasets[domain][split], clip_model, self.device
                )
                self.caches[(domain, split)] = (feats, labels, attrs)
        del clip_model
        torch.cuda.empty_cache()

    def _build_model(self):
        return clip_cbm_orth(
            args=self.args,
            class_names=list(self.classname2id.keys()),
            concept_names=list(self.concept2id.keys()),
            domain_diffs=self.domain_diffs,
        ).to(self.device)

    def _compute_loss(self, cls_preds, labels, reg, model=None, extra=None):
        """Factored out so Phase C's EWC penalty / cumulative-DDO term can
        override this instead of editing the training loop. `model` is
        passed through (unused by the base loss) so remediation subclasses
        can reach model.classifier / model.concept_embeddings without
        threading extra state through the training loop."""
        cls_loss = nn.CrossEntropyLoss()(cls_preds, labels)
        orth_loss = torch.abs(reg).mean() if reg is not None else torch.tensor(0.0, device=self.device)
        return cls_loss + self.args.alpha * orth_loss

    def _train_loop(self, model, optimizer, loader, epochs):
        model.train()
        for _ in range(epochs):
            for feats, labels, _attrs in loader:
                feats = feats.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                _, cls_preds, reg = model.forward_cached(feats)
                loss = self._compute_loss(cls_preds, labels, reg, model=model)
                loss.backward()
                optimizer.step()

    def _eval_all_domains(self, model):
        model.eval()
        accs = {}
        with torch.no_grad():
            for domain in self.domains:
                feats, labels, _ = self.caches[(domain, "test")]
                feats, labels = feats.to(self.device), labels.to(self.device)
                _, cls_preds, _ = model.forward_cached(feats)
                accs[domain] = 100.0 * (cls_preds.argmax(dim=1) == labels).float().mean().item()
        return accs

    def _mechanism_metric(self, model):
        """DDO orthogonality loss against the model's own (fixed-since-
        construction) domain_diffs, using current weights. No data pass -
        diffs/concept_embeddings don't depend on any image batch."""
        model.eval()
        with torch.no_grad():
            reg = model.classifier[1:](model.diffs @ model.concept_embeddings.T)
        return torch.abs(reg).mean().item()

    def run_joint(self, epochs=JOINT_EPOCHS):
        """Non-continual upper bound: train once on all 4 domains pooled."""
        model = self._build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        pooled = ConcatDataset([TensorDataset(*self.caches[(d, "train")]) for d in self.domains])
        loader = DataLoader(pooled, batch_size=self.args.batch_size, shuffle=True)
        self._train_loop(model, optimizer, loader, epochs)
        return self._eval_all_domains(model)

    def _build_stage_loader(self, domain, stage_idx, domain_order):
        """The current stage's training loader. Overridden by Phase C's
        replay remediation to mix in a buffer of prior domains' cached
        features; the base (naive-sequential) behavior is domain i's own
        cached train features only, no replay."""
        return DataLoader(
            TensorDataset(*self.caches[(domain, "train")]), batch_size=self.args.batch_size, shuffle=True
        )

    def run_sequential(self, domain_order, epochs_per_stage=STAGE_EPOCHS, checkpoint_stage0=False):
        """Naive sequential Domain-IL: one persistent model, fresh optimizer
        per stage, no replay of earlier domains' images.

        Reseeds right before model construction so every call - baseline or
        any Phase C remediation, for the same domain_order - starts from an
        identical classifier init and an identical stage-0 batch order. Nothing
        in stage 0 itself (for any remediation here) consumes extra global RNG
        state before this point, so stage 0 comes out bit-identical across
        conditions - the "shared starting point" the plan wanted, without
        needing to share a physical checkpoint across separate runs.
        """
        torch.manual_seed(self.args.seed)
        model = self._build_model()
        pre_training_metric = self._mechanism_metric(model)
        stages = []
        for stage_idx, domain in enumerate(domain_order):
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            loader = self._build_stage_loader(domain, stage_idx, domain_order)
            self._train_loop(model, optimizer, loader, epochs_per_stage)

            per_domain_acc = self._eval_all_domains(model)
            ddo_erosion = self._mechanism_metric(model)
            stages.append({
                "stage": stage_idx,
                "trained_domain": domain,
                "per_domain_test_acc": per_domain_acc,
                "ddo_erosion": ddo_erosion,
            })
            logger.info(
                f"[stage {stage_idx}] trained={domain} "
                f"acc={ {k: round(v, 2) for k, v in per_domain_acc.items()} } "
                f"ddo_erosion={ddo_erosion:.4f}"
            )

            if checkpoint_stage0 and stage_idx == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/phase_b_stage0_shared_start.pth")

            for hook in self.between_stage_hooks:
                hook(self, stage_idx, domain, model, optimizer)

        return stages, pre_training_metric


def compute_acc_bwt(stages, domain_order):
    """ACC = mean final-stage accuracy across all 4 domains.
    BWT = mean, over every domain except the last-trained one, of
    (final accuracy on that domain) - (accuracy on that domain right when
    it finished training). Negative BWT = forgetting."""
    R_final = dict(stages[-1]["per_domain_test_acc"])
    R_diagonal = {d: stages[i]["per_domain_test_acc"][d] for i, d in enumerate(domain_order)}
    ACC = sum(R_final.values()) / len(R_final)
    non_last = domain_order[:-1]
    BWT = sum(R_final[d] - R_diagonal[d] for d in non_last) / len(non_last) if non_last else 0.0
    return ACC, BWT, R_diagonal, R_final


def parse_local_args():
    """Flags specific to this script, parsed separately from args.py's
    get_args() (which consumes the rest of sys.argv)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_test", action="store_true", default=False,
                         help="2 epochs/stage, 1 ordering, writes phase_b_smoke_results.json")
    parser.add_argument("--stage_epochs", type=int, default=STAGE_EPOCHS)
    parser.add_argument("--joint_epochs", type=int, default=JOINT_EPOCHS)
    local_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return local_args


def main():
    local_args = parse_local_args()
    args = get_args()
    args.dataset, args.CBM_type, args.alpha, args.beta = "PACS", "clip_cbm", 1.0, 0.0
    args.batch_size = 64
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    stage_epochs = 2 if local_args.smoke_test else local_args.stage_epochs
    joint_epochs = 2 if local_args.smoke_test else local_args.joint_epochs
    orderings = (
        {"photo_art_cartoon_sketch": DEFAULT_ORDERINGS["photo_art_cartoon_sketch"]}
        if local_args.smoke_test else DEFAULT_ORDERINGS
    )
    output_name = "phase_b_smoke_results.json" if local_args.smoke_test else "phase_b_results.json"

    session = DomainILSession(args)

    logger.info("Running joint/oracle...")
    joint_acc = session.run_joint(joint_epochs)
    logger.info(f"Joint/oracle per-domain acc: {joint_acc}")

    results = {
        "dataset": "PACS",
        "domains": list(PACS_DOMAINS),
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
