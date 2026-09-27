"""Per-image LanCE vs VLM agreement pilot - Gemini side.

For each image in the LanCE pilot set (agreement_pilot_lance.json), shows
Gemini that exact same image plus the full 200-class candidate list, and
asks it to classify + self-report confidence + list concepts.

--constrained mode: instead of writing its own concept phrases, the model
must SELECT from LanCE's actual 312-concept vocabulary (cub_concepts.txt) -
removes the free-text-vs-template-phrase styling mismatch that made the
first (free-form) version's CLIP-similarity concept overlap indistinguishable
from a shuffled baseline. Output becomes exact set membership, comparable to
LanCE's own concepts with no embedding step needed.

Requires GEMINI_API_KEY. Uses gemini-3.1-flash-lite by default - the tier
that actually had free-quota headroom during Plan 09's concept-generation
runs (gemini-3.6-flash capped at 20 requests/day on this key).

Usage:
    python agreement_pilot_query_gemini.py [--sleep SECONDS] [--constrained]
"""
import argparse
import json
import os
import time

from agreement_pilot_common import (
    CUB_DIR,
    IMAGES_DIR,
    build_prompt,
    build_prompt_constrained,
    load_lance_pilot,
    parse_classification_response,
    parse_constrained_response,
)

MODEL = os.environ.get("GEMINI_VLM_MODEL", "gemini-3.1-flash-lite")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument("--constrained", action="store_true", help="select concepts from LanCE's own vocabulary instead of writing free text")
    args = parser.parse_args()
    out_path = "agreement_pilot_gemini_constrained.json" if args.constrained else "agreement_pilot_gemini.json"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY first.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

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

    out_results = []
    for item in pilot["results"]:
        with open(IMAGES_DIR / item["image_path"], "rb") as f:
            image_bytes = f.read()
        parts = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt_text),
        ]

        resp = None
        for attempt in range(6):
            try:
                resp = client.models.generate_content(model=MODEL, contents=parts)
                break
            except Exception as e:
                wait = min(60, args.sleep * (2 ** attempt))
                print(f"{item['image_path']}: attempt {attempt + 1} failed ({e}); retrying in {wait:.0f}s", flush=True)
                time.sleep(wait)
        if resp is None:
            print(f"{item['image_path']}: giving up after repeated failures, skipping", flush=True)
            continue

        raw_text = resp.text or ""
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
        print(f"{item['image_path']}: pred={predicted_class!r} conf={confidence} true={item['true_class']!r} n_invalid={n_invalid}", flush=True)
        time.sleep(args.sleep)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    n_correct = sum(r["vlm_correct"] for r in out_results)
    print(f"Wrote {len(out_results)} results -> {out_path} (Gemini correct: {n_correct}/{len(out_results)})")


if __name__ == "__main__":
    main()
