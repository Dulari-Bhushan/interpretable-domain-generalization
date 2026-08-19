"""One-time DomainNet preprocessing: convert the official per-domain train/test
list files (space-separated "relative/path label", already shipped with the
download - see docs/new_methodology_report.md for the source) into this
project's own manifest format, matching PACS/Office-Home's convention.

Usage:
    python data/DomainNet/prepare_domainnet_dataset.py -data_dir data/DomainNet -save_dir data/DomainNet

Output (written to -save_dir):
    domainnet_classes.txt              - one class name per line, 0-indexed by
                                          line number, derived from the official
                                          label integers (not re-sorted - trusts
                                          the official split's own label<->class
                                          mapping rather than assuming alphabetical
                                          order matches)
    domainnet_<domain>_train.txt       - "relative/path/to/img.jpg,label" per line
    domainnet_<domain>_test.txt        - same format
"""
import os
import argparse

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


def convert_split(data_dir, save_dir, domain, split, label_to_name):
    src_path = os.path.join(data_dir, f"{domain}_{split}.txt")
    out_path = os.path.join(save_dir, f"domainnet_{domain}_{split}.txt")
    n = 0
    with open(src_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            label = int(label)
            class_name = rel_path.split("/")[1]  # "<domain>/<class>/<file>"
            existing = label_to_name.get(label)
            if existing is None:
                label_to_name[label] = class_name
            elif existing != class_name:
                raise ValueError(
                    f"Label {label} maps to both {existing!r} and {class_name!r} - "
                    f"official split's label<->class mapping isn't consistent, stopping."
                )
            fout.write(f"{rel_path},{label}\n")
            n += 1
    print(f"  {domain}/{split}: {n} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required=True, help="Folder containing the official *_train.txt/*_test.txt files")
    parser.add_argument("-save_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    label_to_name = {}
    for domain in DOMAINS:
        print(f"Processing {domain}...")
        convert_split(args.data_dir, args.save_dir, domain, "train", label_to_name)
        convert_split(args.data_dir, args.save_dir, domain, "test", label_to_name)

    n_classes = max(label_to_name.keys()) + 1
    assert n_classes == len(label_to_name), (
        f"expected a dense 0..{n_classes-1} label range, got {len(label_to_name)} distinct labels"
    )
    with open(os.path.join(args.save_dir, "domainnet_classes.txt"), "w") as f:
        for i in range(n_classes):
            f.write(label_to_name[i] + "\n")
    print(f"Wrote domainnet_classes.txt - {n_classes} classes")


if __name__ == "__main__":
    main()
