"""Plan 09: image-grounded concept generation using an open-source VLM
(default Qwen2-VL-7B-Instruct). Meant to be run on the lab GPU server, not
locally — see docs/session_handoff.md for SSH/GPU-etiquette details.

    pip install "transformers>=4.45" accelerate

Usage (on the server, inside the mlgpu conda env):
    python vlm_concepts_generate_opensource.py --gpu 1 [--limit N]
Resumable: already-completed classes (raw/qwen2vl/<class_folder>.json
present) are skipped, so a killed run can just be re-invoked.
"""
import argparse
import os

from vlm_concepts_common import (
    IMAGES_DIR,
    already_done,
    build_prompt,
    load_manifest,
    parse_concept_list,
    save_class_result,
)

MODEL_ID = os.environ.get("OPENSOURCE_VLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
VLM_NAME = "qwen2vl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N classes (pilot run)")
    parser.add_argument("--gpu", type=int, default=1, help="CUDA device index (lab rule: use at most 2 of 3, check nvidia-smi first)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = f"cuda:{args.gpu}"
    print(f"Loading {MODEL_ID} on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

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
        images = [Image.open(IMAGES_DIR / p).convert("RGB") for p in image_paths]

        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": build_prompt(class_name, len(image_paths))})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        concepts = parse_concept_list(raw_text)
        save_class_result(VLM_NAME, class_folder, class_name, image_paths, raw_text, concepts)
        print(f"{class_folder}: {len(concepts)} concepts")


if __name__ == "__main__":
    main()
