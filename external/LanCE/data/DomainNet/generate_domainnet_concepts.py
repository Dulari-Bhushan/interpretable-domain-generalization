"""Generates domainnet_concepts.txt from domainnet_classes.txt.

Deliberate, stated simplification: PACS (70 concepts/7 classes) and
Office-Home (257/65) both used hand-written, class-specific concepts.
DomainNet's 345 classes make that impractical to hand-write at the same
density for what this experiment actually needs. Component 1's DomainNet
run tests a mathematical property (sequential training matches joint
training exactly) that holds regardless of what the concepts are - it does
not depend on concept quality the way a classification-accuracy experiment
would. So this generates 4 generic, template-based concepts per class
instead of hand-curating them, and says so plainly rather than presenting
them as equivalent in quality to PACS/Office-Home's concept banks. If a
later experiment needs DomainNet classification *accuracy* to be
meaningful (not just the exactness property), these should be replaced
with real, class-specific concepts first.

Usage:
    python data/DomainNet/generate_domainnet_concepts.py
"""
import os

TEMPLATES = [
    "the overall shape of a {}",
    "the typical color of a {}",
    "a distinctive part of a {}",
    "the texture or surface of a {}",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "domainnet_classes.txt"), "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    lines = []
    for cls in classes:
        readable = cls.replace("_", " ")
        for template in TEMPLATES:
            lines.append(template.format(readable))

    out_path = os.path.join(here, "domainnet_concepts.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path} - {len(lines)} concepts for {len(classes)} classes "
          f"({len(TEMPLATES)} generic templates/class)")


if __name__ == "__main__":
    main()
