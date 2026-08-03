"""One-time PACS preprocessing: scan raw domain/class image folders (or an
official kfold split manifest, if the mirror ships one) and write flat
metadata files consumed by pacs_data.py.

Usage:
    python data/PACS/prepare_pacs_dataset.py -data_dir <path to folder containing
        photo/, art_painting/, cartoon/, sketch/> -save_dir data/PACS

Output (written to -save_dir):
    pacs_classes.txt              - one class name per line, 0-indexed by line number
                                     (unlike CUB's classes.txt, which is 1-indexed and
                                     needs a -1 correction at read time - avoided here
                                     on purpose since we control this file end-to-end)
    pacs_<domain>_train.txt       - "relative/path/to/img.jpg,label" per line
    pacs_<domain>_test.txt        - same format, 20% held out per class (seed=0)
                                     if no official split manifest is found
"""
import os
import random
import argparse
from collections import defaultdict

CANONICAL_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
# Folder-name variants seen across different PACS mirrors/redistributions.
DOMAIN_ALIASES = {
    "photo": ["photo"],
    "art_painting": ["art_painting", "art painting", "art-painting"],
    "cartoon": ["cartoon"],
    "sketch": ["sketch"],
}
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
TEST_FRACTION = 0.2
SPLIT_SEED = 0


def resolve_domain_dir(data_dir, domain):
    for alias in DOMAIN_ALIASES[domain]:
        candidate = os.path.join(data_dir, alias)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a folder for domain '{domain}' under {data_dir} "
        f"(tried aliases: {DOMAIN_ALIASES[domain]})"
    )


def scan_domain(domain_dir):
    """Returns dict[class_name] -> list of relative image paths under domain_dir."""
    class_to_paths = defaultdict(list)
    class_folders = sorted(
        f for f in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, f))
    )
    for cls in class_folders:
        cls_dir = os.path.join(domain_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(IMG_EXTENSIONS):
                class_to_paths[cls.lower()].append(os.path.join(cls, fname).replace("\\", "/"))
    return class_to_paths


def find_official_split_dir(data_dir):
    """Official PACS mirrors commonly ship a 'pacs_label' folder with
    <domain>_{train,test}_kfold.txt manifests ('relative/path label', 1-indexed,
    space-separated). Auto-detect it; return None if not present so we fall
    back to generating our own split.
    """
    for name in ("pacs_label", "PACS_label", "Train val splits and h5py files pre-read"):
        candidate = os.path.join(data_dir, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def load_official_split(split_dir, domain, split, classname2id):
    """Returns list of (relative_path, 0-indexed_label), or None if the
    expected manifest file isn't present under split_dir."""
    manifest = os.path.join(split_dir, f"{domain}_{split}_kfold.txt")
    if not os.path.exists(manifest):
        return None
    entries = []
    with open(manifest, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label_1idx = line.split()
            cls = path.split("/")[0].lower()
            if cls not in classname2id:
                raise ValueError(f"Unknown class '{cls}' in {manifest}: {line}")
            entries.append((path, classname2id[cls]))
    return entries


def stratified_split(class_to_paths, classname2id, seed=SPLIT_SEED, test_fraction=TEST_FRACTION):
    rng = random.Random(seed)
    train_entries, test_entries = [], []
    for cls, paths in class_to_paths.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        n_test = max(1, round(len(paths) * test_fraction))
        test_paths, train_paths = paths[:n_test], paths[n_test:]
        label = classname2id[cls]
        train_entries.extend((p, label) for p in train_paths)
        test_entries.extend((p, label) for p in test_paths)
    return train_entries, test_entries


def write_manifest(path, entries):
    with open(path, "w") as f:
        for rel_path, label in entries:
            f.write(f"{rel_path},{label}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare PACS metadata files for pacs_data.py")
    parser.add_argument("-data_dir", required=True,
                         help="Folder directly containing photo/, art_painting/, cartoon/, sketch/")
    parser.add_argument("-save_dir", default="data/PACS", help="Where to write the metadata files")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Scan all 4 domains first so pacs_classes.txt reflects the union of
    # classes actually present (should be all 7, but don't assume).
    domain_dirs = {d: resolve_domain_dir(args.data_dir, d) for d in CANONICAL_DOMAINS}
    domain_class_paths = {d: scan_domain(domain_dirs[d]) for d in CANONICAL_DOMAINS}

    all_classes = sorted(set().union(*[set(cp.keys()) for cp in domain_class_paths.values()]))
    classname2id = {cls: i for i, cls in enumerate(all_classes)}
    with open(os.path.join(args.save_dir, "pacs_classes.txt"), "w") as f:
        for cls in all_classes:
            f.write(f"{cls}\n")
    print(f"Classes ({len(all_classes)}): {all_classes}")

    official_split_dir = find_official_split_dir(args.data_dir)
    if official_split_dir:
        print(f"Found official split manifest dir: {official_split_dir}")

    total_counts = {}
    for domain in CANONICAL_DOMAINS:
        train_entries = test_entries = None
        if official_split_dir:
            train_entries = load_official_split(official_split_dir, domain, "train", classname2id)
            test_entries = load_official_split(official_split_dir, domain, "test", classname2id)

        if train_entries is None or test_entries is None:
            train_entries, test_entries = stratified_split(domain_class_paths[domain], classname2id)

        write_manifest(os.path.join(args.save_dir, f"pacs_{domain}_train.txt"), train_entries)
        write_manifest(os.path.join(args.save_dir, f"pacs_{domain}_test.txt"), test_entries)

        per_class_counts = defaultdict(int)
        for _p, label in train_entries + test_entries:
            per_class_counts[label] += 1
        total = len(train_entries) + len(test_entries)
        total_counts[domain] = total
        print(f"{domain}: {total} images total "
              f"({len(train_entries)} train / {len(test_entries)} test) "
              f"| per-class: {dict(sorted(per_class_counts.items()))}")

    grand_total = sum(total_counts.values())
    print(f"\nGrand total: {grand_total} images across {len(CANONICAL_DOMAINS)} domains "
          f"(expected ~9,991 for standard PACS)")


if __name__ == "__main__":
    main()
