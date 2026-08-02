"""Phase A - closed-world descriptor assumption test.

DDO's benefit depends on the 200-item, pre-enumerated domain-descriptor
pool (prompts/prompt200new.py) containing something close to the *actual*
test domain. Their own Table 2/8 shows this with one binary relevant/
irrelevant split. This script extends it to a continuous dose-response:
rank all 200 descriptors by similarity to the true "painting" domain
shift, then retrain DDO with progressively more of the *most similar*
descriptors excluded from the pool - i.e. simulate a continual setting
where the incoming domain is progressively less anticipated.

Image embeddings are loaded from Phase 0's cache (identical images, only
the descriptor pool changes) - only the domain-diff text embeddings are
recomputed per condition, so this is fast (~30-60s per condition).
"""
import os
import sys
import json
import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args
from model.cbm_models import clip_cbm_orth
from data.CUB.cub_data import Processed_CUB_Dataset
from prompts.prompt200new import source_text_prompts, target_text_prompts
from cache_utils import get_or_build_feature_cache

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results")
TRUE_DOMAIN_PROMPT = "a painting of a {}."
EXCLUDE_TOP_K_CONDITIONS = [0, 20, 50, 100, 150, 180]
EPOCHS = 50  # match Phase 0 exactly so exclude_top_0 is a direct control check


def load_class_names(classes_txt_path):
    with open(classes_txt_path, "r") as f:
        return [x.split(" ")[1][4:].replace("_", " ").lower().rstrip() for x in f.readlines()]


def class_averaged_diff(clip_model, src_prompt, tgt_prompt, class_names, device):
    src_texts = clip.tokenize([src_prompt.format(c) for c in class_names]).to(device)
    tgt_texts = clip.tokenize([tgt_prompt.format(c) for c in class_names]).to(device)
    with torch.no_grad():
        src_emb = clip_model.encode_text(src_texts).float()
        tgt_emb = clip_model.encode_text(tgt_texts).float()
    src_emb = src_emb / src_emb.norm(dim=-1, keepdim=True)
    tgt_emb = tgt_emb / tgt_emb.norm(dim=-1, keepdim=True)
    diff = tgt_emb - src_emb
    diff = diff / diff.norm(dim=-1, keepdim=True)
    return diff.mean(dim=0)  # (feat_dim,) - one direction representing this descriptor's domain shift


def rank_descriptors_by_similarity(clip_model, class_names, device):
    true_dir = class_averaged_diff(clip_model, source_text_prompts[0], TRUE_DOMAIN_PROMPT, class_names, device)
    true_dir = true_dir / true_dir.norm()

    sims = []
    for p in target_text_prompts:
        d = class_averaged_diff(clip_model, source_text_prompts[0], p, class_names, device)
        d = d / d.norm()
        sims.append((p, torch.dot(d, true_dir).item()))

    sims.sort(key=lambda x: x[1], reverse=True)  # most similar to "painting" first
    return sims


def train_one_condition(args, descriptor_pool, class_names, train_feats, train_labels, train_attrs,
                          source_feats, source_labels, source_attrs, target_feats, target_labels, target_attrs):
    # Build a dataset object purely to get domain_diffs computed against this
    # descriptor subset (text-only, no images touched) + classname2id/concept2id.
    data_root = os.path.join(args.data_dir, "CUB/CUB_200_2011/images")
    ds = Processed_CUB_Dataset(
        args, data_root=data_root, split="train", meta_root="data/CUB",
        attr_name="CUBpath2attr.pkl", src_dm_texts=source_text_prompts, tgt_dm_texts=descriptor_pool,
    )

    model = clip_cbm_orth(
        args=args, class_names=list(ds.classname2id.keys()),
        concept_names=list(ds.concept2id.keys()), domain_diffs=ds.domain_diffs,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cls_loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(train_feats, train_labels, train_attrs), batch_size=args.batch_size, shuffle=True)
    target_loader = DataLoader(TensorDataset(target_feats, target_labels, target_attrs), batch_size=args.batch_size, shuffle=False)

    best_target_acc = 0.0
    for _epoch in range(EPOCHS):
        model.train()
        for feats, labels, _attrs in train_loader:
            feats, labels = feats.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            _, cls_preds, reg_loss = model.forward_cached(feats)
            loss = cls_loss_fn(cls_preds, labels) + args.alpha * torch.abs(reg_loss).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels, _attrs in target_loader:
                feats, labels = feats.to(args.device), labels.to(args.device)
                _, cls_preds, _ = model.forward_cached(feats)
                correct += (cls_preds.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        best_target_acc = max(best_target_acc, acc)

    return best_target_acc


def main():
    args = get_args()
    args.dataset, args.alpha, args.class_avg_concept, args.CBM_type = "CUB", 1.0, True, "clip_cbm"
    args.batch_size = 64  # match Phase 0's runs exactly
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_model, _ = clip.load(args.CLIP_type, device=args.device)
    for p in clip_model.parameters():
        p.requires_grad = False

    class_names_raw = load_class_names(os.path.join(args.data_dir, "CUB/CUB_200_2011/classes.txt"))

    print("Ranking 200 descriptors by similarity to the true 'painting' domain shift...")
    ranked = rank_descriptors_by_similarity(clip_model, class_names_raw, args.device)
    print(f"Most similar: {ranked[0]}")
    print(f"Least similar: {ranked[-1]}")

    cache_prefix = f"CUB_{args.CLIP_type}".replace("/", "-")
    train_feats, train_labels, train_attrs = get_or_build_feature_cache(f"{cache_prefix}_train", None, clip_model, args.device)
    source_feats, source_labels, source_attrs = get_or_build_feature_cache(f"{cache_prefix}_source_test", None, clip_model, args.device)
    target_feats, target_labels, target_attrs = get_or_build_feature_cache(f"{cache_prefix}_target_test", None, clip_model, args.device)

    results = []
    for exclude_k in EXCLUDE_TOP_K_CONDITIONS:
        pool = [p for p, _sim in ranked[exclude_k:]]  # drop the exclude_k MOST similar descriptors
        remaining_sims = [sim for _p, sim in ranked[exclude_k:]]
        mean_sim = sum(remaining_sims) / len(remaining_sims)

        print(f"\n=== exclude_top_{exclude_k} (pool size {len(pool)}, mean similarity {mean_sim:.4f}) ===")
        acc = train_one_condition(
            args, pool, class_names_raw,
            train_feats, train_labels, train_attrs,
            source_feats, source_labels, source_attrs,
            target_feats, target_labels, target_attrs,
        )
        print(f"Best target accuracy: {acc:.2f}%")
        results.append({"exclude_top_k": exclude_k, "pool_size": len(pool), "mean_similarity": mean_sim, "target_acc": acc})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_a_results.json")
    with open(out_path, "w") as f:
        json.dump({"true_domain_prompt": TRUE_DOMAIN_PROMPT, "ranked_top5": ranked[:5], "ranked_bottom5": ranked[-5:], "conditions": results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
