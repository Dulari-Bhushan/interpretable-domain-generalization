"""One-time extraction: pull the Midjourney (target-domain) images out of the
GenImage zip for whichever ImageNet classes actually survived in this
particular download.

Context (see results/phase_f4_genimage_alignment.md for full detail): the
GenImage Midjourney folder is downloaded as one part of a multi-part Google
Drive archive. Only the final part was fetched here, so most of train/ai,
train/nature, and val/ai are *listed* in the zip's central directory (which
spans the whole multi-part archive) but their actual bytes are in the
missing earlier parts and fail to extract ("bad zipfile offset"). What does
extract cleanly from this part: val/ai for 155 of the 1,000 ImageNet
classes (~6 images/class, 928 images total) - real labeled training photos
(train/nature) were not recoverable, which is why Phase F4 is an alignment-
score test (like Phase F1/F3), not a full baseline-vs-DDO training run.

Python's zipfile module refuses this archive outright ("zipfiles that span
multiple disks are not supported"), so this shells out to the `unzip` CLI,
which handles it (with a warning) as long as the requested entries' data
happens to be present in this part.

Class-index -> name is resolved via imagenet_class_index.json (fetched from
a standard public mirror, saved alongside this script) - verified
empirically against actual images before trusting it (index 8 -> checked
the image -> shows a hen; index 999 -> checked the image -> shows toilet
paper; both match the standard convention).

Usage:
    python data/GenImage/extract_genimage_subset.py -zip_path <path to imagenet_midjourney.zip>
"""
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-zip_path", required=True)
    parser.add_argument("-save_dir", default="data/GenImage/genimage_raw")
    args = parser.parse_args()

    # Only val/ai actually has recoverable data in a typical partial download
    # of this archive - train/ai, train/nature, and most of val/ai are listed
    # but not present. Extracting the whole val/ai/* glob and letting
    # whichever classes are actually present come through is simpler and more
    # honest than pre-selecting a class list that assumes what will succeed.
    cmd = ["unzip", "-o", args.zip_path, "imagenet_midjourney/val/ai/*", "-d", args.save_dir]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode not in (0, 1):  # unzip returns 1 for partial-archive warnings, still usable
        print(result.stderr[-2000:])
        raise RuntimeError(f"unzip failed with code {result.returncode}")

    ai_dir = os.path.join(args.save_dir, "imagenet_midjourney", "val", "ai")
    files = os.listdir(ai_dir)
    classes = sorted(set(int(f.split("_midjourney_")[0]) for f in files))
    print(f"\nExtracted {len(files)} images across {len(classes)} classes to {ai_dir}")
    print(f"Class index range: {min(classes)}-{max(classes)}")


if __name__ == "__main__":
    main()
