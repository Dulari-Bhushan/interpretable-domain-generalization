"""Phase F4 - domain-shift alignment check on GenImage/Midjourney (Pillar 2,
temporal-novelty variant, third independent dataset).

Same methodology as Phase F1 (EuroSAT) and Phase F3 (Defactify/MS-COCO-AI):
    visual_shift  = mean_image_embedding(Midjourney images of class c)
                    - text_embedding("a photo of a {class c}.")
    textual_shift = text_embedding("a Midjourney-generated image of a {class c}.")
                    - text_embedding("a photo of a {class c}.")
    alignment     = cosine_similarity(visual_shift, textual_shift)
averaged over however many classes have images available.

Why GenImage/Midjourney as a third data point: it's the dataset the
original planning docs identified as the ideal test for temporal novelty
before being dropped as impractical (data-access issues - see
planning/01-lance-failure-mode-analysis-plan.md). Revisited here once the
user located and downloaded the Midjourney subset directly.

Data availability note (see results write-up for full detail): the download
is one part of a multi-part Google Drive archive (only the final volume was
fetched) - most of train/ai and train/nature, and most of val/ai, are
listed in the central directory but their actual bytes are not present in
this part ("bad zipfile offset" on extraction). val/ai (Midjourney images)
for 155 of the 1000 ImageNet classes did extract successfully and cleanly
(verified: class-index -> name mapping checked against two actual images by
eye - index 8 shows a hen, index 999 shows toilet paper, matching the
standard ImageNet class-index convention). Real, correctly-labeled source
photos (train/nature) were not recoverable from this part, so - as in
Phase F1 - this uses CLIP text as the photo-domain reference rather than
real photos (unlike Phase F3, which did have real matched photos via
Defactify/MS-COCO-AI).
"""
import os
import sys
import json
import re

import torch
import clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
GENIMAGE_AI_DIR = "data/GenImage/genimage_raw/imagenet_midjourney/val/ai"
CLASS_INDEX_PATH = "data/GenImage/imagenet_class_index.json"
PAPER_ALIGNMENT_RANGE = (0.90, 0.99)
EUROSAT_F1_MEAN_ALIGNMENT = 0.324
DEFACTIFY_F3_MEAN_ALIGNMENT = 0.037  # per-class-controlled version


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(args.CLIP_type, device=args.device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    with open(CLASS_INDEX_PATH) as f:
        class_index = json.load(f)  # str(idx) -> [synset, name]

    files = os.listdir(GENIMAGE_AI_DIR)
    files_by_class = {}
    for f in files:
        m = re.match(r"(\d+)_midjourney", f)
        idx = int(m.group(1))
        files_by_class.setdefault(idx, []).append(f)

    print(f"{len(files_by_class)} classes available, {len(files)} images total")

    def display_name(idx):
        return class_index[str(idx)][1].replace("_", " ")

    def encode_images(paths):
        from PIL import Image
        feats = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                batch = paths[i:i + 64]
                tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(args.device)
                f = clip_model.encode_image(tensors).float()
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    def text_embed(prompt):
        with torch.no_grad():
            tok = clip.tokenize([prompt]).to(args.device)
            emb = clip_model.encode_text(tok).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu()

    per_class_alignment = {}
    for idx, fnames in sorted(files_by_class.items()):
        name = display_name(idx)
        paths = [os.path.join(GENIMAGE_AI_DIR, f) for f in fnames]
        img_feats = encode_images(paths)
        mean_img = img_feats.mean(dim=0)
        mean_img = mean_img / mean_img.norm()

        text_photo = text_embed(f"a photo of a {name}.")
        text_mj = text_embed(f"a Midjourney-generated image of a {name}.")

        visual_shift = mean_img - text_photo
        visual_shift = visual_shift / visual_shift.norm()
        textual_shift = text_mj - text_photo
        textual_shift = textual_shift / textual_shift.norm()

        alignment = torch.dot(visual_shift, textual_shift).item()
        per_class_alignment[name] = {"class_idx": idx, "n_images": len(fnames), "alignment": alignment}

    mean_alignment = sum(v["alignment"] for v in per_class_alignment.values()) / len(per_class_alignment)
    print(f"\nMean alignment across {len(per_class_alignment)} GenImage/Midjourney classes: {mean_alignment:.4f}")
    print(f"Paper's own range: {PAPER_ALIGNMENT_RANGE}")
    print(f"Phase F1 EuroSAT: {EUROSAT_F1_MEAN_ALIGNMENT}")
    print(f"Phase F3 Defactify (per-class): {DEFACTIFY_F3_MEAN_ALIGNMENT}")

    results = {
        "clip_type": args.CLIP_type,
        "n_classes": len(per_class_alignment),
        "per_class_alignment": per_class_alignment,
        "mean_alignment": mean_alignment,
        "paper_alignment_range": PAPER_ALIGNMENT_RANGE,
        "eurosat_f1_mean_alignment": EUROSAT_F1_MEAN_ALIGNMENT,
        "defactify_f3_mean_alignment": DEFACTIFY_F3_MEAN_ALIGNMENT,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_f4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
