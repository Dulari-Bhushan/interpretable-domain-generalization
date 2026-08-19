"""Component 2 - self-diagnosing domain grounding.

DDO's orthogonality regularizer (clip_cbm_orth) consumes a domain_diffs
tensor of shape (num_directions, num_classes, feat_dim) built entirely from
text (see utils.get_domain_text_embs): domain_diffs[i] = normalize(
text_embedding(target_template) - text_embedding(source_template)), per
class. Phase F1/F3/F4 measured that this text-only guess about a new
domain's visual shift is unreliable outside CLIP's comfort zone (alignment
0.04-0.32 against the paper's own claimed 0.90-0.99). This module makes
that measurement part of the method itself: a cheap diagnostic run on a
small probe of real images from the arriving domain, deciding whether to
trust the text-only domain_diffs (current LanCE behavior) or fall back to
measuring domain_diffs directly from the probe images.

Nothing in clip_cbm_orth's forward pass requires domain_diffs come from
text - it only ever does `self.diffs @ self.concept_embeddings.T`, and the
leading (num_directions) dimension is averaged away in the orthogonality
loss (`torch.abs(reg).mean()` in every training script in this project).
So an image-grounded domain_diffs with num_directions=1 is a drop-in
replacement, no model changes needed.
"""
import torch
import clip


def _encode_text_template(clip_model, template, class_names, device):
    """text_emb[c] = normalize(CLIP text embedding of template.format(class_names[c]))."""
    texts = clip.tokenize([template.format(c) for c in class_names]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(texts).float()
    return emb / emb.norm(dim=-1, keepdim=True)


def _mean_image_embeddings_by_class(clip_model, images_by_class, class_names, device, preprocess=None):
    """images_by_class[c] = list of already-preprocessed image tensors (C,H,W)
    OR list of PIL images (if preprocess is given). Returns (num_classes,
    feat_dim), L2-normalized mean embedding per class."""
    means = []
    with torch.no_grad():
        for c in class_names:
            imgs = images_by_class[c]
            if preprocess is not None:
                imgs = [preprocess(im) for im in imgs]
            batch = torch.stack(imgs).to(device)
            emb = clip_model.encode_image(batch).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean_emb = emb.mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()
            means.append(mean_emb)
    return torch.stack(means, dim=0)  # (num_classes, feat_dim)


def compute_alignment_score(clip_model, source_images_by_class, target_images_by_class,
                             class_names, source_template, target_template, device,
                             preprocess=None):
    """Phase F3's alignment-score methodology (real matched photos on both
    sides - the cleanest of the three F1/F3/F4 variants), generalized to run
    on a small probe set rather than a full held-out sample.

    source_images_by_class may be None - Phase F1/F4's adaptation for domains
    with no matched real "photo" dataset available (e.g. EuroSAT: there's no
    "ground photo of a forest/highway/..." dataset on hand). When None, the
    source side uses CLIP's own text embedding of source_template as the
    photo-domain reference point instead of real images, exactly as Phase F1
    documented and justified (CLIP's near-unity image-text alignment for
    ordinary photo domains). Flagged here, not hidden, the same way Phase F1
    flagged it: this trades the real-photo formula's cleanliness (Phase F3's
    version) for applicability to domains without a matched photo dataset.

    Returns (per_class_alignment: dict[str, float], mean_alignment: float).
    """
    src_text_emb = _encode_text_template(clip_model, source_template, class_names, device)
    if source_images_by_class is None:
        src_img_emb = src_text_emb  # Phase F1's text-proxy adaptation
    else:
        src_img_emb = _mean_image_embeddings_by_class(clip_model, source_images_by_class, class_names, device, preprocess)
    tgt_img_emb = _mean_image_embeddings_by_class(clip_model, target_images_by_class, class_names, device, preprocess)
    tgt_text_emb = _encode_text_template(clip_model, target_template, class_names, device)

    per_class_alignment = {}
    for i, c in enumerate(class_names):
        visual_shift = tgt_img_emb[i] - src_img_emb[i]
        visual_shift = visual_shift / visual_shift.norm()
        textual_shift = tgt_text_emb[i] - src_text_emb[i]
        textual_shift = textual_shift / textual_shift.norm()
        per_class_alignment[c] = torch.dot(visual_shift, textual_shift).item()

    mean_alignment = sum(per_class_alignment.values()) / len(per_class_alignment)
    return per_class_alignment, mean_alignment


def build_image_grounded_domain_diffs(clip_model, source_images_by_class, target_images_by_class,
                                       class_names, device, preprocess=None):
    """Replaces the text-only domain_diffs with a single real-image-measured
    direction per class: normalize(mean(target probe embeddings) -
    mean(source probe embeddings)). Shape (1, num_classes, feat_dim) -
    same convention clip_cbm_orth already expects (num_directions dim first).
    """
    src_img_emb = _mean_image_embeddings_by_class(clip_model, source_images_by_class, class_names, device, preprocess)
    tgt_img_emb = _mean_image_embeddings_by_class(clip_model, target_images_by_class, class_names, device, preprocess)
    diff = tgt_img_emb - src_img_emb
    diff = diff / diff.norm(dim=-1, keepdim=True)
    return diff.unsqueeze(0).to(device)  # (1, num_classes, feat_dim)


def build_image_grounded_domain_diffs_persample(clip_model, source_images_by_class, target_images_by_class,
                                                 class_names, device, preprocess=None):
    """Diversity-preserving alternative to build_image_grounded_domain_diffs.

    The single-mean-direction fallback (Component 2's first validation run,
    results/component2_self_diagnosing_grounding.md) underperformed baseline
    - one direction estimated from a small probe is far noisier than the
    204 diverse (if individually imperfect) text directions it replaces.
    This keeps every probe image as its OWN direction relative to the
    class's mean source embedding, instead of collapsing the whole probe to
    one mean - shape (K, num_classes, feat_dim) where K = images/class in
    the target probe (same convention as text_domain_diffs' (204, C, F),
    just grounded in real images instead of template phrasings).

    Requires every class's target probe to have exactly the same number of
    images K (true by construction wherever this project draws probes -
    each class's probe is capped to the same PROBE_PER_CLASS).
    """
    src_img_emb = _mean_image_embeddings_by_class(clip_model, source_images_by_class, class_names, device, preprocess)
    per_class_dirs = []
    with torch.no_grad():
        for i, c in enumerate(class_names):
            imgs = target_images_by_class[c]
            if preprocess is not None:
                imgs = [preprocess(im) for im in imgs]
            batch = torch.stack(imgs).to(device)
            emb = clip_model.encode_image(batch).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)  # (K, feat_dim)
            diffs = emb - src_img_emb[i]  # (K, feat_dim)
            diffs = diffs / diffs.norm(dim=-1, keepdim=True)
            per_class_dirs.append(diffs)
    # (num_classes, K, feat_dim) -> (K, num_classes, feat_dim)
    return torch.stack(per_class_dirs, dim=0).permute(1, 0, 2).contiguous().to(device)


def blend_domain_diffs(*domain_diffs_tensors):
    """Concatenates domain_diffs tensors along the num_directions dimension
    - e.g. blend_domain_diffs(text_domain_diffs, grounded_domain_diffs) keeps
    every text direction AND adds the image-grounded one(s), rather than
    replacing text with images outright. All inputs must share the same
    (num_classes, feat_dim) trailing shape."""
    return torch.cat(domain_diffs_tensors, dim=0)


def diagnose_and_build_domain_diffs(clip_model, source_images_by_class, target_images_by_class,
                                     class_names, source_template, target_template,
                                     text_domain_diffs, threshold, device, preprocess=None):
    """Orchestrator: run the diagnostic; if the text-only guess is
    trustworthy (mean alignment >= threshold), keep the existing text-based
    domain_diffs (zero extra real images needed beyond the probe used to
    check). If not, fall back to the image-grounded version built from the
    same probe.

    Returns dict with keys: per_class_alignment, mean_alignment, threshold,
    trusted (bool), domain_diffs (the tensor to actually train with).
    """
    per_class_alignment, mean_alignment = compute_alignment_score(
        clip_model, source_images_by_class, target_images_by_class,
        class_names, source_template, target_template, device, preprocess,
    )
    trusted = mean_alignment >= threshold
    if trusted:
        domain_diffs = text_domain_diffs
    else:
        domain_diffs = build_image_grounded_domain_diffs(
            clip_model, source_images_by_class, target_images_by_class, class_names, device, preprocess,
        )
    return {
        "per_class_alignment": per_class_alignment,
        "mean_alignment": mean_alignment,
        "threshold": threshold,
        "trusted": trusted,
        "domain_diffs": domain_diffs,
    }
