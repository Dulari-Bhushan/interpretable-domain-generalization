"""Plan 09: Grounding-DINO -> CLIP concept discovery, the non-generative
alternative pipeline. Grounding DINO detects anatomical-part regions in each
image from a fixed set of text queries; each cropped region is then matched
by CLIP image-text similarity against a candidate concept vocabulary (built
by vlm_concepts_build_candidate_vocab.py -- CLIP has no decoder, so it can
only pick from an existing vocabulary, not write new phrases).

Must run where CLIP + torch + transformers are available (the lab GPU
server) -- see docs/session_handoff.md for SSH details.
    pip install "transformers>=4.45"   # already required by the open-source
                                        # VLM leg; grounding-dino support
                                        # needs a reasonably recent version

Usage (on the server, inside the mlgpu conda env):
    python vlm_concepts_generate_dino_clip.py --gpu 1 [--limit N]
Resumable: already-completed classes (raw/dino_clip/<class_folder>.json
present) are skipped.
"""
import argparse
import os

from vlm_concepts_common import (
    IMAGES_DIR,
    already_done,
    load_manifest,
    save_class_result,
)

DINO_MODEL_ID = os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
VLM_NAME = "dino_clip"

# A fixed, generic set of anatomical-part queries -- not class-specific, since
# Grounding DINO is prompted once per image, not per class.
PART_QUERIES = [
    "bird", "bill", "wing", "eye", "tail", "leg", "head", "feather",
    "breast", "belly", "crown", "throat", "back", "neck",
]

TOP_K_PER_BOX = 2
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2


def load_candidate_vocab():
    from pathlib import Path

    path = Path(__file__).parent / "vlm_concepts" / "clip_candidate_vocab.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run vlm_concepts_build_candidate_vocab.py first.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N classes (pilot run)")
    parser.add_argument("--gpu", type=int, default=1, help="CUDA device index (lab rule: use at most 2 of 3, check nvidia-smi first)")
    args = parser.parse_args()

    import clip
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = f"cuda:{args.gpu}"

    print(f"Loading Grounding DINO ({DINO_MODEL_ID}) on {device} ...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
    dino_model.eval()
    dino_text_query = ". ".join(PART_QUERIES) + "."

    print("Loading CLIP ViT-L/14 ...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()

    vocab = load_candidate_vocab()
    with torch.no_grad():
        vocab_tokens = clip.tokenize(vocab, truncate=True).to(device)
        vocab_emb = clip_model.encode_text(vocab_tokens).float()
        vocab_emb = vocab_emb / vocab_emb.norm(dim=-1, keepdim=True)
    print(f"Candidate vocabulary: {len(vocab)} phrases")

    manifest = load_manifest()
    items = list(manifest.items())
    if args.limit:
        items = items[: args.limit]

    for class_folder, info in items:
        if already_done(VLM_NAME, class_folder):
            print(f"skip (done): {class_folder}")
            continue

        class_name = info["class_name"]
        image_paths = info["image_paths"]

        class_concepts = []
        n_boxes_total = 0
        for image_path in image_paths:
            image = Image.open(IMAGES_DIR / image_path).convert("RGB")

            dino_inputs = dino_processor(images=image, text=dino_text_query, return_tensors="pt").to(device)
            with torch.no_grad():
                dino_outputs = dino_model(**dino_inputs)
            results = dino_processor.post_process_grounded_object_detection(
                dino_outputs,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )[0]

            boxes = results["boxes"]
            if len(boxes) == 0:
                continue
            n_boxes_total += len(boxes)

            crops = []
            for box in boxes.tolist():
                x0, y0, x1, y1 = [max(0, int(v)) for v in box]
                x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
                crops.append(clip_preprocess(image.crop((x0, y0, x1, y1))))
            crop_batch = torch.stack(crops).to(device)

            with torch.no_grad():
                crop_emb = clip_model.encode_image(crop_batch).float()
            crop_emb = crop_emb / crop_emb.norm(dim=-1, keepdim=True)

            sim = crop_emb @ vocab_emb.T  # [n_boxes, n_vocab]
            topk = sim.topk(k=min(TOP_K_PER_BOX, sim.shape[1]), dim=1)
            for box_idx in range(sim.shape[0]):
                for rank in range(topk.indices.shape[1]):
                    class_concepts.append(vocab[topk.indices[box_idx, rank].item()])

        # Dedup while preserving first-seen order (same convention as
        # vlm_concepts_pool.py's cross-class dedup, applied here per-class).
        deduped = []
        seen = set()
        for c in class_concepts:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        save_class_result(
            VLM_NAME, class_folder, class_name, image_paths, raw_text=None, concepts=deduped,
            model=f"grounding-dino:{DINO_MODEL_ID}+clip:ViT-L/14",
        )
        print(f"{class_folder}: {n_boxes_total} boxes across {len(image_paths)} images -> {len(deduped)} unique concepts")


if __name__ == "__main__":
    main()
