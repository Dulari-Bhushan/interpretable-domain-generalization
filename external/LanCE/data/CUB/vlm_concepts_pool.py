"""Plan 09: pool per-class VLM concept outputs into one deduplicated concept
bank file, matching the style of the existing hand-written/LLM concept banks
(external/LanCE/data/CUB/cub_concepts*.txt) so it's a drop-in --concept_file.

Usage:
    python vlm_concepts_pool.py --vlm claude
    python vlm_concepts_pool.py --vlm gemini
    python vlm_concepts_pool.py --vlm qwen2vl
"""
import argparse
import json
import re
from pathlib import Path

CUB_DIR = Path(__file__).parent
RAW_DIR = CUB_DIR / "vlm_concepts" / "raw"


def normalize(phrase):
    p = phrase.lower().strip()
    p = re.sub(r"^(a|an|the)\s+", "", p)
    p = re.sub(r"[^a-z0-9\s]", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", required=True, choices=["claude", "gemini", "qwen2vl"])
    args = parser.parse_args()

    raw_dir = RAW_DIR / args.vlm
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No raw outputs found in {raw_dir} — run the generation script for '{args.vlm}' first.")

    seen = set()
    pooled = []
    n_classes = 0
    n_concepts_raw = 0
    for f in files:
        data = json.loads(f.read_text())
        n_classes += 1
        for phrase in data["concepts"]:
            n_concepts_raw += 1
            key = normalize(phrase)
            if not key or key in seen:
                continue
            seen.add(key)
            pooled.append(phrase.strip())

    out_path = CUB_DIR / f"cub_concepts_{args.vlm}_grounded.txt"
    out_path.write_text("\n".join(pooled) + "\n")

    print(f"{args.vlm}: {n_classes} classes processed, {n_concepts_raw} raw concepts -> {len(pooled)} unique -> {out_path}")
    if n_classes < 200:
        print(f"WARNING: only {n_classes}/200 classes have raw output — this is a partial pool, not the full bank.")


if __name__ == "__main__":
    main()
