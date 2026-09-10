"""Plan 09: sample a fixed set of real training images per CUB class.

Used as the shared input to every VLM concept-generation script (Claude,
Gemini, open-source), so any quality difference between them is attributable
to the model, not to which images happened to be sampled.
"""
import json
import random
from pathlib import Path

CUB_DIR = Path(__file__).parent
TRAIN_SPLIT = CUB_DIR / "cub_train.txt"
IMAGES_DIR = CUB_DIR / "CUB_200_2011" / "images"
OUT_DIR = CUB_DIR / "vlm_concepts"
OUT_MANIFEST = OUT_DIR / "sampled_images.json"

N_PER_CLASS = 5
SEED = 42


def main():
    by_class = {}
    with open(TRAIN_SPLIT) as f:
        for line in f:
            path = line.split(",", 1)[0]
            class_folder = path.split("/")[0]  # e.g. "001.Black_footed_Albatross"
            by_class.setdefault(class_folder, []).append(path)

    rng = random.Random(SEED)
    manifest = {}
    for class_folder, paths in sorted(by_class.items()):
        class_id, class_name_raw = class_folder.split(".", 1)
        class_name = class_name_raw.replace("_", " ")
        sample = sorted(rng.sample(paths, min(N_PER_CLASS, len(paths))))
        for p in sample:
            full = IMAGES_DIR / p
            if not full.exists():
                raise FileNotFoundError(f"missing image: {full}")
        manifest[class_folder] = {
            "class_id": int(class_id),
            "class_name": class_name,
            "image_paths": sample,
        }

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Sampled {N_PER_CLASS} images for {len(manifest)} classes -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
