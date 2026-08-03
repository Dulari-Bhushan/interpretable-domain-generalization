"""Phase F3 - domain-shift alignment check on Defactify/MS-COCO-AI (Pillar 2,
temporal-novelty variant).

Phase F1 (EuroSAT) tested modality scarcity - EuroSAT actually predates CLIP,
so that result is about satellite imagery being rare in captioned web
photos, not about temporal novelty. This phase tests genuine temporal
novelty instead: all 5 generators here were released well after both CLIP's
training cutoff (~2020/2021) and GPT-3.5's knowledge cutoff (~Sept 2021 -
what LanCE's 200-descriptor list was generated from):
  - Stable Diffusion 2.1 - Dec 2022
  - Stable Diffusion XL  - Jul 2023
  - DALL-E 3             - Oct 2023
  - Midjourney v6        - Dec 2023
  - Stable Diffusion 3   - 2024

Improvement over Phase F1's methodology: this dataset gives REAL matched
photo images (1,500 MS COCO photos, Label_B=0) alongside each generator's AI
images, built from the same underlying COCO captions/scenes. So the visual
side of the alignment score uses actual photo embeddings this time, not a
CLIP-text stand-in for "what a photo would look like" - closing the
adaptation gap flagged in Phase F1's write-up.

No discrete class labels are provided (only free-text COCO-style captions),
so this computes one global alignment score per generator rather than
per-class:
    visual_shift  = mean_image_embedding(generator's AI images)
                    - mean_image_embedding(real COCO photos)
    textual_shift = text_embedding("a {generator} generated image.")
                    - text_embedding("a real photograph.")
    alignment     = cosine_similarity(visual_shift, textual_shift)

No training run - same "cheap, minimal setup" shape as Phase F1.
"""
import os
import sys
import json
import random
from collections import defaultdict

import torch
import clip
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from args import get_args

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "results",
)
HF_CACHE_DIR = "C:/hfc"
SAMPLES_PER_SOURCE = 300
SEED = 0
MIN_PER_CLASS = 8  # keep a class only if both real and this generator have >= this many tagged samples

# The dataset ships free-text captions, not official COCO category
# annotations - this is a first-match keyword heuristic over the 80
# standard COCO categories, an approximation, not ground truth.
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def first_matching_category(caption):
    cl = caption.lower()
    for cat in COCO_CATEGORIES:
        if cat in cl or f"{cat}s" in cl:
            return cat
    return None

# Label_B -> (display name, release date), confirmed against the dataset card
GENERATOR_INFO = {
    1: ("Stable Diffusion 2.1 generated image", "Dec 2022"),
    2: ("Stable Diffusion XL generated image", "Jul 2023"),
    3: ("Stable Diffusion 3 generated image", "2024"),
    4: ("DALL-E 3 generated image", "Oct 2023"),
    5: ("Midjourney v6 generated image", "Dec 2023"),
}
PAPER_ALIGNMENT_RANGE = (0.90, 0.99)
EUROSAT_F1_MEAN_ALIGNMENT = 0.324  # results/phase_f1_results.json, for direct comparison


def encode_images(clip_model, preprocess, pil_images, device, batch_size=64):
    feats = []
    with torch.no_grad():
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
            f = clip_model.encode_image(tensors).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def text_embed(clip_model, device, prompt):
    with torch.no_grad():
        tok = clip.tokenize([prompt]).to(device)
        emb = clip_model.encode_text(tok).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu()


def main():
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(args.CLIP_type, device=args.device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    print("Loading Defactify/MS-COCO-AI dataset (cached locally)...")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", cache_dir=HF_CACHE_DIR)["validation"]
    label_b = ds["Label_B"]
    captions = ds["Caption"]

    # ---------- Part 1: global alignment (mixed-content domain shift) ----------
    rng = random.Random(SEED)
    idx_by_label = {}
    for lbl in [0] + list(GENERATOR_INFO.keys()):
        idxs = [i for i, l in enumerate(label_b) if l == lbl]
        rng.shuffle(idxs)
        idx_by_label[lbl] = idxs[:SAMPLES_PER_SOURCE]

    print(f"[Global] sampling {SAMPLES_PER_SOURCE} images per source (real + {len(GENERATOR_INFO)} generators)")
    print("[Global] encoding real COCO photos...")
    real_feats_g = encode_images(clip_model, preprocess, [ds[i]["Image"] for i in idx_by_label[0]], args.device)
    real_centroid_g = real_feats_g.mean(dim=0)
    real_centroid_g = real_centroid_g / real_centroid_g.norm()
    text_real_emb = text_embed(clip_model, args.device, "a real photograph.")

    global_per_generator = {}
    for lbl, (name, release_date) in GENERATOR_INFO.items():
        print(f"[Global] encoding {name} ({release_date})...")
        gen_feats = encode_images(clip_model, preprocess, [ds[i]["Image"] for i in idx_by_label[lbl]], args.device)
        gen_centroid = gen_feats.mean(dim=0)
        gen_centroid = gen_centroid / gen_centroid.norm()

        visual_shift = gen_centroid - real_centroid_g
        visual_shift = visual_shift / visual_shift.norm()
        text_gen_emb = text_embed(clip_model, args.device, f"a {name}.")
        textual_shift = text_gen_emb - text_real_emb
        textual_shift = textual_shift / textual_shift.norm()

        alignment = torch.dot(visual_shift, textual_shift).item()
        global_per_generator[name] = {"label_b": lbl, "release_date": release_date, "alignment": alignment}
        print(f"  global alignment: {alignment:.4f}")

    global_mean_alignment = sum(v["alignment"] for v in global_per_generator.values()) / len(global_per_generator)

    # ---------- Part 2: per-class-controlled alignment ----------
    # Global alignment mixes many different COCO scene types into one domain-shift
    # vector - content diversity unrelated to the photo-vs-generated distinction could
    # dilute the signal. This tags each image via a keyword match against the 80
    # standard COCO categories (approximation, not ground truth - the dataset ships
    # captions, not category annotations) and computes alignment within each
    # sufficiently-represented category, then averages - directly analogous to how
    # Phase F1 controlled for class on EuroSAT.
    print("\n[Per-class] tagging captions against 80 COCO categories...")
    tags = [first_matching_category(c) for c in captions]

    idx_by_label_category = defaultdict(lambda: defaultdict(list))
    for i, (lbl, tag) in enumerate(zip(label_b, tags)):
        if tag is not None and lbl in ([0] + list(GENERATOR_INFO.keys())):
            idx_by_label_category[lbl][tag].append(i)

    real_categories = idx_by_label_category[0]
    print(f"[Per-class] {len(real_categories)} categories tagged among real photos")

    # cache per (label, category) mean image embedding, computed once
    def category_centroid(lbl, cat, cap=60):
        idxs = idx_by_label_category[lbl].get(cat, [])[:cap]
        if len(idxs) < MIN_PER_CLASS:
            return None, len(idxs)
        feats = encode_images(clip_model, preprocess, [ds[i]["Image"] for i in idxs], args.device)
        c = feats.mean(dim=0)
        return c / c.norm(), len(idxs)

    print("[Per-class] precomputing real-photo centroids/text embeddings per category (shared across generators)...")
    real_centroid_by_cat, real_text_by_cat = {}, {}
    for cat in real_categories:
        c, _n = category_centroid(0, cat)
        if c is not None:
            real_centroid_by_cat[cat] = c
            real_text_by_cat[cat] = text_embed(clip_model, args.device, f"a real photograph of a {cat}.")

    per_class_per_generator = {}
    for lbl, (name, release_date) in GENERATOR_INFO.items():
        print(f"[Per-class] {name}...")
        class_alignments = {}
        for cat, real_c in real_centroid_by_cat.items():
            gen_c, n_gen = category_centroid(lbl, cat)
            if gen_c is None:
                continue
            visual_shift = gen_c - real_c
            visual_shift = visual_shift / visual_shift.norm()
            text_gen_c = text_embed(clip_model, args.device, f"a {name} of a {cat}.")
            textual_shift = text_gen_c - real_text_by_cat[cat]
            textual_shift = textual_shift / textual_shift.norm()
            class_alignments[cat] = torch.dot(visual_shift, textual_shift).item()

        if class_alignments:
            mean_class_alignment = sum(class_alignments.values()) / len(class_alignments)
        else:
            mean_class_alignment = None
        per_class_per_generator[name] = {
            "label_b": lbl,
            "release_date": release_date,
            "n_categories_used": len(class_alignments),
            "per_category_alignment": class_alignments,
            "mean_alignment": mean_class_alignment,
        }
        print(f"  {len(class_alignments)} categories used, mean alignment: {mean_class_alignment}")

    valid = [v["mean_alignment"] for v in per_class_per_generator.values() if v["mean_alignment"] is not None]
    per_class_overall_mean = sum(valid) / len(valid) if valid else None

    print(f"\n=== Summary ===")
    print(f"Global (mixed-content) mean alignment:     {global_mean_alignment:.4f}")
    print(f"Per-class-controlled mean alignment:       {per_class_overall_mean}")
    print(f"Paper's own range (sketch/sculpture/painting): {PAPER_ALIGNMENT_RANGE}")
    print(f"Phase F1 EuroSAT (modality scarcity) mean alignment: {EUROSAT_F1_MEAN_ALIGNMENT}")

    results = {
        "clip_type": args.CLIP_type,
        "global": {
            "samples_per_source": SAMPLES_PER_SOURCE,
            "per_generator": global_per_generator,
            "mean_alignment": global_mean_alignment,
        },
        "per_class_controlled": {
            "min_per_class": MIN_PER_CLASS,
            "per_generator": per_class_per_generator,
            "mean_alignment": per_class_overall_mean,
        },
        "paper_alignment_range": PAPER_ALIGNMENT_RANGE,
        "eurosat_f1_mean_alignment": EUROSAT_F1_MEAN_ALIGNMENT,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "phase_f3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
