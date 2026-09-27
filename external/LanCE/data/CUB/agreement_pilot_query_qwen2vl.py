"""Per-image LanCE vs VLM agreement pilot - Qwen2-VL side.

For each image in the LanCE pilot set (agreement_pilot_lance.json), shows
Qwen2-VL-7B-Instruct that exact same image plus the full 200-class candidate
list, and asks it to classify + self-report confidence + list concepts.
Meant to run on the lab GPU server - see docs/session_handoff.md.

--constrained mode: instead of writing its own concept phrases, the model
must SELECT from LanCE's actual 312-concept vocabulary (cub_concepts.txt) -
removes the free-text-vs-template-phrase styling mismatch that made the
first (free-form) version's CLIP-similarity concept overlap indistinguishable
from a shuffled baseline.

Usage:
    python agreement_pilot_query_qwen2vl.py --gpu 1 [--constrained]
"""
import argparse
import json
import os

from agreement_pilot_common import (
    CUB_DIR,
    IMAGES_DIR,
    build_prompt,
    build_prompt_constrained,
    load_lance_pilot,
    parse_classification_response,
    parse_constrained_response,
)

MODEL_ID = os.environ.get("OPENSOURCE_VLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--constrained", action="store_true", help="select concepts from LanCE's own vocabulary instead of writing free text")
    args = parser.parse_args()
    out_path = "agreement_pilot_qwen2vl_constrained.json" if args.constrained else "agreement_pilot_qwen2vl.json"

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = f"cuda:{args.gpu}"
    pilot = load_lance_pilot()
    class_names = pilot["class_names"]
    if args.constrained:
        concept_names = [
            line.rstrip("\n") for line in open(CUB_DIR / "cub_concepts.txt", encoding="utf-8") if line.strip()
        ]
        prompt_text = build_prompt_constrained(class_names, concept_names)
    else:
        concept_names = None
        prompt_text = build_prompt(class_names)

    print(f"Loading {MODEL_ID} on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    out_results = []
    for item in pilot["results"]:
        image = Image.open(IMAGES_DIR / item["image_path"]).convert("RGB")
        content = [{"type": "image"}, {"type": "text", "text": prompt_text}]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

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

        if args.constrained:
            predicted_class, confidence, concepts, n_invalid = parse_constrained_response(raw_text, class_names, concept_names)
        else:
            predicted_class, confidence, concepts = parse_classification_response(raw_text, class_names)
            n_invalid = None
        out_results.append(
            {
                "image_path": item["image_path"],
                "true_class": item["true_class"],
                "vlm_pred_class": predicted_class,
                "vlm_confidence": confidence,
                "vlm_concepts": concepts,
                "vlm_n_invalid_concepts": n_invalid,
                "vlm_correct": predicted_class == item["true_class"] if predicted_class else False,
                "raw_response": raw_text,
            }
        )
        print(f"{item['image_path']}: pred={predicted_class!r} conf={confidence} true={item['true_class']!r} n_invalid={n_invalid}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    n_correct = sum(r["vlm_correct"] for r in out_results)
    print(f"Wrote {len(out_results)} results -> {out_path} (Qwen2-VL correct: {n_correct}/{len(out_results)})")


if __name__ == "__main__":
    main()
