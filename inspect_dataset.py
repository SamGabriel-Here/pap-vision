"""Answer one question before training: can patients be recovered from filenames?

Everything about this dataset's evaluation hinges on splitting by patient rather
than by image, and that is only possible if the filenames carry a patient or
slide identifier. Run this the moment the data finishes downloading:

    python inspect_dataset.py --data-dir data/

It needs nothing but the standard library — run it before installing torch.

It prints what the filenames look like, tries a set of candidate patterns, and
tells you either the exact `train.py` command to use or that no grouping is
possible and why that matters.
"""

import argparse
import pathlib
import re
from collections import Counter

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["HSIL", "LSIL", "NILM", "SCC"]

# One group cannot be split three ways, so a pattern yielding fewer than this
# many groups has told us nothing usable.
MIN_USABLE_GROUPS = 2

# Ordered most-specific first; the first pattern that groups well enough wins.
CANDIDATE_PATTERNS = [
    (r"(?i)(?:patient|case|subject)[-_ ]?(\d+)", "explicit patient/case number"),
    (r"(?i)(?:slide|sl)[-_ ]?(\d+)", "explicit slide number"),
    (r"^(\d+)[-_]", "leading digits before a separator"),
    (r"^(\d+)", "leading digits"),
    (r"^([^_]+)_", "everything before the first underscore"),
    (r"^([^-]+)-", "everything before the first hyphen"),
    (r"^(.+?)[-_ ]\d+\.[^.]+$", "prefix before a trailing image number"),
]


def evaluate_pattern(filenames, pattern):
    """How well does this pattern group these filenames?"""
    groups = []
    for name in filenames:
        match = re.search(pattern, name)
        if match:
            groups.append(match.group(1) if match.groups() else match.group(0))

    counts = Counter(groups)
    matched = len(groups)

    return {
        "pattern": pattern,
        "match_ratio": matched / len(filenames) if filenames else 0.0,
        "groups": len(counts),
        "images": len(filenames),
        "max_per_group": max(counts.values()) if counts else 0,
        # The whole point. If every group holds exactly one image the pattern
        # has not grouped anything, and a "grouped" split is just a per-image
        # split wearing a disguise.
        "groups_anything": bool(counts) and len(counts) < matched,
    }


def rank_patterns(filenames, min_match_ratio=0.95, max_images_per_group=50):
    """Usable patterns in specificity order, best first.

    Usable means three things, and the third is easy to forget: the pattern
    matches nearly every filename, it collapses multiple images into shared
    groups, and it does not collapse *too* far. A pattern like "everything
    before the first underscore" applied to `patient_001_a.jpg` captures
    "patient" for every file and yields a single group — maximal grouping,
    zero information, and a split that cannot be made at all. Ranking by
    fewest-groups would rate that the best candidate available.
    """
    scored = []
    for pattern, description in CANDIDATE_PATTERNS:
        stats = evaluate_pattern(filenames, pattern)
        stats["description"] = description

        density = stats["images"] / stats["groups"] if stats["groups"] else 0
        stats["images_per_group"] = round(density, 2)
        stats["usable"] = (
            stats["match_ratio"] >= min_match_ratio
            and stats["groups_anything"]
            and stats["groups"] >= MIN_USABLE_GROUPS
            and density <= max_images_per_group
        )
        scored.append(stats)

    # Specificity order, i.e. the order the candidates are declared in. An
    # explicit "patient_012" beats a generic prefix rule even when the generic
    # rule produces a tidier-looking number of groups.
    return [stats for stats in scored if stats["usable"]], scored


def collect(data_dir):
    """Filenames per class folder."""
    root = pathlib.Path(data_dir)
    if not root.is_dir():
        raise SystemExit(f"No such directory: {root}")

    per_class = {}
    for label in CLASS_NAMES:
        folder = root / label
        if not folder.is_dir():
            continue
        per_class[label] = sorted(
            path.name for path in folder.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )

    if not per_class:
        found = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "nothing"
        raise SystemExit(
            f"Found no class folders under {root}.\n"
            f"Expected directories named: {', '.join(CLASS_NAMES)}\n"
            f"Found instead: {found}"
        )
    return per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    per_class = collect(args.data_dir)
    everything = [name for names in per_class.values() for name in names]

    print(f"{len(everything)} images across {len(per_class)} class folders\n")
    for label, names in per_class.items():
        print(f"  {label:<5} {len(names):>4} images")
        for name in names[:args.samples]:
            print(f"        {name}")
    print()

    missing = [label for label in CLASS_NAMES if label not in per_class]
    if missing:
        print(f"!! missing class folders: {', '.join(missing)}\n")

    usable, scored = rank_patterns(everything)

    print("candidate patient-id patterns\n")
    for stats in scored:
        mark = "ok  " if stats["usable"] else "no  "
        print(f"  {mark}{stats['description']}")
        print(f"      {stats['pattern']}")
        print(f"      matched {stats['match_ratio']:>6.1%}   "
              f"{stats['groups']} groups for {stats['images']} images   "
              f"max {stats['max_per_group']} per group")
    print()

    if usable:
        best = usable[0]
        ratio = best["images"] / best["groups"]
        print(f"Recommended: {best['description']}")
        print(f"  {best['groups']} patients, {ratio:.2f} images each on average.\n")
        print("Train with:\n")
        print(f"  python train.py --data-dir {args.data_dir} \\")
        print(f"      --patient-regex '{best['pattern']}'\n")
        print("Read the split summary it prints before letting it train.")
    else:
        print("No pattern recovered patient identifiers.\n")
        print("Every candidate either failed to match or produced one group per")
        print("image, which is not grouping at all. If these filenames genuinely")
        print("carry no patient or slide id, a clean split is impossible from")
        print("this data alone, and any metric you produce will be inflated by")
        print("images of the same patient sitting on both sides of the split.\n")
        print("Options, in order of preference:\n")
        print("  1. Check the dataset download for a metadata file (CSV, XLSX)")
        print("     mapping images to patients — this set was published with a")
        print("     paper, so a mapping may exist outside the filenames.")
        print("  2. Pass your own pattern if you can see structure this script")
        print("     missed:  python train.py --patient-regex 'YOUR_REGEX'")
        print("  3. Accept the leak explicitly and say so in the README:\n")
        print(f"       python train.py --data-dir {args.data_dir} --group-by image\n")
        print("     This stamps a leakage warning into docs/metrics.json. Those")
        print("     numbers are an upper bound, not a result — never quote them")
        print("     without the caveat attached.")


if __name__ == "__main__":
    main()
