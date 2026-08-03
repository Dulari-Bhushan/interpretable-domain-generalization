"""One-time Office-Home preprocessing: scan the 4 raw domain/class image
folders and write flat metadata files consumed by office_home_data.py.

Usage:
    python data/OfficeHome/prepare_office_home_dataset.py -data_dir <path to
        folder containing Art/, Clipart/, Product/, "Real World"/>
        -save_dir data/OfficeHome

Output (written to -save_dir):
    office_home_classes.txt         - one class name per line, 0-indexed by
                                       line number
    office_home_<domain>_train.txt  - "relative/path/to/img.jpg,label" per line
    office_home_<domain>_test.txt   - same format, 20% held out per class
                                       (seed=0) - the official release has no
                                       train/test split manifest (only source
                                       URLs in imagelist.txt/ImageInfo.csv),
                                       so this generates one, same convention
                                       as prepare_pacs_dataset.py.
"""
import os
import random
import argparse
from collections import defaultdict

# domain key -> actual folder name (Office-Home's "Real World" folder has a
# space, so it needs its own key without one)
DOMAIN_FOLDERS = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
}
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
TEST_FRACTION = 0.2
SPLIT_SEED = 0


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
                class_to_paths[cls].append(os.path.join(cls, fname).replace("\\", "/"))
    return class_to_paths


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
    parser = argparse.ArgumentParser(description="Prepare Office-Home metadata files for office_home_data.py")
    parser.add_argument("-data_dir", required=True,
                         help='Folder directly containing Art/, Clipart/, Product/, "Real World"/')
    parser.add_argument("-save_dir", default="data/OfficeHome", help="Where to write the metadata files")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    domain_dirs = {}
    for key, folder in DOMAIN_FOLDERS.items():
        path = os.path.join(args.data_dir, folder)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Expected domain folder '{folder}' under {args.data_dir}, not found")
        domain_dirs[key] = path

    domain_class_paths = {key: scan_domain(d) for key, d in domain_dirs.items()}

    # all_classes holds the raw folder names (e.g. "Alarm_Clock") - used to
    # look up which folder each image lives under. office_home_classes.txt
    # instead gets a cleaned, human-readable display form ("alarm clock"),
    # same convention as CUB's classes.txt - id ordering (line i = id i) is
    # preserved between the two.
    all_classes = sorted(set().union(*[set(cp.keys()) for cp in domain_class_paths.values()]))
    classname2id = {cls: i for i, cls in enumerate(all_classes)}
    with open(os.path.join(args.save_dir, "office_home_classes.txt"), "w") as f:
        for cls in all_classes:
            f.write(f"{cls.replace('_', ' ').lower()}\n")
    print(f"Classes ({len(all_classes)}): {all_classes[:5]}... (+{len(all_classes) - 5} more)")

    total_counts = {}
    for key in DOMAIN_FOLDERS:
        train_entries, test_entries = stratified_split(domain_class_paths[key], classname2id)
        write_manifest(os.path.join(args.save_dir, f"office_home_{key}_train.txt"), train_entries)
        write_manifest(os.path.join(args.save_dir, f"office_home_{key}_test.txt"), test_entries)

        total = len(train_entries) + len(test_entries)
        total_counts[key] = total
        n_classes_present = len(domain_class_paths[key])
        print(f"{key}: {total} images total "
              f"({len(train_entries)} train / {len(test_entries)} test), "
              f"{n_classes_present} classes present")

    grand_total = sum(total_counts.values())
    print(f"\nGrand total: {grand_total} images across {len(DOMAIN_FOLDERS)} domains "
          f"(expected ~15,588 for standard Office-Home)")


if __name__ == "__main__":
    main()
