"""Plan 09: image-grounded concept generation using Gemini.

Requires GEMINI_API_KEY (https://aistudio.google.com/apikey — note this is
separate from a consumer Gemini app subscription, billed per the API's own
quota/pricing).
    pip install google-genai

Usage:
    python vlm_concepts_generate_gemini.py [--limit N] [--sleep SECONDS]
Resumable: already-completed classes (raw/gemini/<class_folder>.json present)
are skipped, so a killed/rate-limited run can just be re-invoked.
"""
import argparse
import os
import time

from vlm_concepts_common import (
    already_done,
    build_prompt,
    load_image_bytes,
    load_manifest,
    parse_concept_list,
    save_class_result,
)

MODEL = os.environ.get("GEMINI_VLM_MODEL", "gemini-3.6-flash")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N classes (pilot run)")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds to sleep between API calls")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY first (create one at https://aistudio.google.com/apikey).")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    manifest = load_manifest()
    items = list(manifest.items())
    if args.limit:
        items = items[: args.limit]

    for class_folder, info in items:
        if already_done("gemini", class_folder):
            print(f"skip (done): {class_folder}")
            continue

        class_name = info["class_name"]
        image_paths = info["image_paths"]
        parts = [
            types.Part.from_bytes(data=load_image_bytes(p), mime_type="image/jpeg") for p in image_paths
        ]
        parts.append(types.Part.from_text(text=build_prompt(class_name, len(image_paths))))

        resp = client.models.generate_content(model=MODEL, contents=parts)
        raw_text = resp.text or ""
        concepts = parse_concept_list(raw_text)
        save_class_result("gemini", class_folder, class_name, image_paths, raw_text, concepts)
        print(f"{class_folder}: {len(concepts)} concepts")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
