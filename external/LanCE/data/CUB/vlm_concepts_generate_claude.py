"""Plan 09: image-grounded concept generation using Claude.

Requires ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys).
    pip install anthropic

Usage:
    python vlm_concepts_generate_claude.py [--limit N] [--sleep SECONDS]
Resumable: already-completed classes (raw/claude/<class_folder>.json present)
are skipped, so a killed/rate-limited run can just be re-invoked.
"""
import argparse
import os
import time

from vlm_concepts_common import (
    already_done,
    build_prompt,
    image_to_base64,
    load_manifest,
    parse_concept_list,
    save_class_result,
)

MODEL = os.environ.get("CLAUDE_VLM_MODEL", "claude-sonnet-5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N classes (pilot run)")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds to sleep between API calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ANTHROPIC_API_KEY first (create one at https://console.anthropic.com/settings/keys)."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    manifest = load_manifest()
    items = list(manifest.items())
    if args.limit:
        items = items[: args.limit]

    for class_folder, info in items:
        if already_done("claude", class_folder):
            print(f"skip (done): {class_folder}")
            continue

        class_name = info["class_name"]
        image_paths = info["image_paths"]
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_to_base64(p)},
            }
            for p in image_paths
        ]
        content.append({"type": "text", "text": build_prompt(class_name, len(image_paths))})

        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = "".join(block.text for block in resp.content if block.type == "text")
        concepts = parse_concept_list(raw_text)
        save_class_result("claude", class_folder, class_name, image_paths, raw_text, concepts, model=MODEL)
        print(f"{class_folder}: {len(concepts)} concepts")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
