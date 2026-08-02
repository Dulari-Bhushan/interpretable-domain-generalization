"""Precompute and cache frozen CLIP image embeddings.

CLIP never updates during LanCE training (all its parameters have
requires_grad=False), so its output for a given image is identical on
every epoch. The released code recomputes it anyway, on every image, on
every epoch - the dominant cost of a training run. This module computes
it once per dataset split and caches to disk.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_or_build_feature_cache(name, dataset, clip_model, device, cache_dir="embeddings_cache", batch_size=128):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path)
        return cached["features"], cached["labels"], cached["attr_labels"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    all_feats, all_labels, all_attrs = [], [], []
    clip_model.eval()
    with torch.no_grad():
        for images, labels, attr_labels in tqdm(loader, desc=f"Caching {name}"):
            images = images.to(device)
            feats = clip_model.encode_image(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            all_attrs.append(attr_labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    attrs = torch.cat(all_attrs, dim=0)
    torch.save({"features": features, "labels": labels, "attr_labels": attrs}, cache_path)
    return features, labels, attrs
