"""Same training procedure as main.py, but on precomputed/cached CLIP embeddings.

Reuses the exact dataset construction, model architecture, DDO loss, and
optimizer setup from main.py/data/__init__.py/model/cbm_models.py - the
only difference is that the (frozen, never-updated) CLIP image encoder
runs once per image via cache_utils instead of once per image per epoch.
Log format matches main.py's so downstream tooling (graphs, results docs)
doesn't need to care which script produced a given log.
"""
import os
import time
import json
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from args import get_args
from model.cbm_models import clip_cbm_orth, clip_mlp
from data import get_dataset_classes
from cache_utils import get_or_build_feature_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training_cached.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CachedTrainingSession:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args.device = self.device
        logger.info(f"Using device: {self.device}")
        self._init_components()
        self.best_metrics = {"source_acc": 0.0, "target_acc": 0.0, "epoch": -1}

    def _init_components(self):
        # Build the original datasets (correct labels/preprocessing/domain_diffs),
        # discard their raw-image loaders - we only need the dataset objects.
        train_dataset, _, source_test_dataset, _, target_test_dataset, _ = get_dataset_classes(self.args)
        logger.info(f"Dataset {self.args.dataset} loaded")

        model_factory = {"clip_cbm": clip_cbm_orth, "cliplp": clip_mlp}
        model_class = model_factory[self.args.CBM_type]
        self.model = model_class(
            args=self.args,
            class_names=list(train_dataset.classname2id.keys()),
            concept_names=list(train_dataset.concept2id.keys()),
            domain_diffs=train_dataset.domain_diffs,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {self.args.CBM_type}")
        logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

        cache_prefix = f"{self.args.dataset}_{self.args.CLIP_type}".replace("/", "-")
        train_feats, train_labels, train_attrs = get_or_build_feature_cache(
            f"{cache_prefix}_train", train_dataset, self.model.clip_model, self.device
        )
        source_feats, source_labels, source_attrs = get_or_build_feature_cache(
            f"{cache_prefix}_source_test", source_test_dataset, self.model.clip_model, self.device
        )
        target_feats, target_labels, target_attrs = get_or_build_feature_cache(
            f"{cache_prefix}_target_test", target_test_dataset, self.model.clip_model, self.device
        )
        logger.info(
            f"Cached features - train: {len(train_feats):,}, "
            f"source test: {len(source_feats):,}, target test: {len(target_feats):,}"
        )

        self.train_loader = DataLoader(
            TensorDataset(train_feats, train_labels, train_attrs),
            batch_size=self.args.batch_size, shuffle=True,
        )
        self.source_test_loader = DataLoader(
            TensorDataset(source_feats, source_labels, source_attrs),
            batch_size=self.args.batch_size, shuffle=False,
        )
        self.target_test_loader = DataLoader(
            TensorDataset(target_feats, target_labels, target_attrs),
            batch_size=self.args.batch_size, shuffle=False,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.loss_fns = {"cls": nn.CrossEntropyLoss(), "concept": nn.BCEWithLogitsLoss()}

    def _run_epoch(self, loader, train: bool, desc: str):
        self.model.train() if train else self.model.eval()
        total, correct = 0, 0
        total_loss = cls_loss = concept_loss = orth_loss = torch.tensor(0.0)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for features, labels, attr_labels in loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                attr_labels = attr_labels.to(self.device, non_blocking=True).float()

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                concept_preds, cls_preds, reg_loss = self.model.forward_cached(features)

                cls_loss = self.loss_fns["cls"](cls_preds, labels)
                concept_loss = self.loss_fns["concept"](concept_preds, attr_labels)
                orth_loss = torch.abs(reg_loss).mean() if reg_loss is not None else torch.tensor(0.0, device=self.device)
                total_loss = cls_loss + self.args.alpha * orth_loss + self.args.beta * concept_loss

                if train:
                    total_loss.backward()
                    self.optimizer.step()

                total += features.size(0)
                correct += (cls_preds.argmax(dim=1) == labels).sum().item()

        return {
            "acc": 100 * correct / total,
            "loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "concept_loss": concept_loss.item(),
            "orth_loss": orth_loss.item() if torch.is_tensor(orth_loss) else orth_loss,
        }

    def run(self):
        logger.info("\n" + "=" * 60)
        logger.info(f"Starting Training for {self.args.epochs} epochs")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Model: {self.args.CBM_type}")
        logger.info(f"Alpha: {self.args.alpha}, Beta: {self.args.beta}")
        logger.info("=" * 60 + "\n")

        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True, desc="train")
            source_metrics = self._run_epoch(self.source_test_loader, train=False, desc="source")
            target_metrics = self._run_epoch(self.target_test_loader, train=False, desc="target")

            if target_metrics["acc"] > self.best_metrics["target_acc"]:
                self.best_metrics.update(
                    {"source_acc": source_metrics["acc"], "target_acc": target_metrics["acc"], "epoch": epoch}
                )
                if self.args.save_model:
                    os.makedirs("checkpoints", exist_ok=True)
                    run_tag = f"{self.args.dataset}_{self.args.CBM_type}_alpha{self.args.alpha}"
                    torch.save(self.model.state_dict(), f"checkpoints/best_{run_tag}.pth")

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1} Summary: Time: {epoch_time:.2f}s | "
                f"Train Acc: {train_metrics['acc']:.2f}% | "
                f"Source Acc: {source_metrics['acc']:.2f}% | "
                f"Target Acc: {target_metrics['acc']:.2f}% | "
                f"Best Target Acc: {self.best_metrics['target_acc']:.2f}% @ Epoch {self.best_metrics['epoch'] + 1}"
            )

        logger.info("\n" + "=" * 60)
        logger.info("Training Completed")
        logger.info(f"Best Source Accuracy: {self.best_metrics['source_acc']:.2f}%")
        logger.info(f"Best Target Accuracy: {self.best_metrics['target_acc']:.2f}%")
        logger.info("=" * 60)


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    session = CachedTrainingSession(args)
    session.run()
