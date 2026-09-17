"""Plan 09 follow-up: subsample a concept bank down to a fixed size, to test
whether a bank's downstream-accuracy edge is about concept *quality* or just
concept *count*. Same anchor size (263, Qwen2-VL's actual bank size) as the
CLIP-similarity size-normalization in vlm_concepts_eval_quality.py, so this
reuses the same experimental logic for the downstream-accuracy axis.

Usage:
    python vlm_concepts_subsample_bank.py --bank cub_concepts_gemini_grounded.txt --size 263 --seed 0
Writes <bank_stem>_sub<size>_seed<seed>.txt in the same directory.
"""
import argparse
import random
from pathlib import Path

CUB_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    bank_path = CUB_DIR / args.bank
    phrases = [line.strip() for line in bank_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(phrases) < args.size:
        raise SystemExit(f"{args.bank} has only {len(phrases)} concepts, fewer than requested size {args.size}")

    rng = random.Random(args.seed)
    sampled = rng.sample(phrases, args.size)

    out_path = CUB_DIR / f"{bank_path.stem}_sub{args.size}_seed{args.seed}.txt"
    out_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")
    print(f"{args.bank}: sampled {args.size}/{len(phrases)} (seed {args.seed}) -> {out_path}")


if __name__ == "__main__":
    main()
