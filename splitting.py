"""Leakage-safe dataset splitting.

The Mendeley LBC set is 962 images drawn from just 61 slides — roughly sixteen
images per slide, not the two-per-patient the published description implies
(see DATASET.md). A random per-image split therefore puts images from the same
slide on both sides of the train/test boundary almost every time, and the model
only has to recognise a slide's staining and illumination signature rather than
its pathology. The resulting accuracy is inflated and meaningless, and nothing
about the training run looks wrong.

Measured on this dataset, that is worth +0.257 macro-F1 of pure illusion, and it
turns a model that never detects carcinoma into one that appears to detect it
nine times in ten.

So splitting happens by *group* (slide or patient), never by image, and the
result is asserted disjoint rather than assumed to be.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

# Matches a leading run of digits, e.g. "0142_a.jpg" -> "0142". This is a guess
# at the dataset's naming scheme and is deliberately overridable, because
# guessing wrong here is the single most expensive mistake in this pipeline.
DEFAULT_PATIENT_PATTERN = r"^(\d+)"


# Two is the smallest number of folds that can hold out anything at all.
MIN_FOLDS = 2


class GroupingError(Exception):
    """Patient/slide identifiers could not be recovered from the filenames."""


@dataclass(frozen=True)
class Record:
    path: str
    label: str
    group: str


def infer_group(filename, pattern=DEFAULT_PATIENT_PATTERN):
    """Extract a patient/slide identifier from a filename, or None.

    The identifier is lowercased. Real datasets are inconsistent about case —
    Mendeley LBC ships `scc_1` and `SCC_3` in the same folder — and treating
    those as different slides would split one slide across train and test,
    which is precisely the leak this module exists to prevent.
    """
    match = re.search(pattern, filename)
    if match is None:
        return None
    captured = match.group(1) if match.groups() else match.group(0)
    return captured.strip().rstrip("_").lower()


def build_records(files, pattern=DEFAULT_PATIENT_PATTERN, min_match_ratio=0.95):
    """Turn (filename, label) pairs into Records carrying a group identifier.

    Raises GroupingError when the pattern fails to match often enough. Failing
    loudly is the point: silently falling back to one-group-per-image would
    produce a leaky split that still trains, still reports metrics, and is
    entirely wrong.
    """
    if not files:
        raise GroupingError("No input files were provided")

    matched = [(name, label, infer_group(name, pattern)) for name, label in files]
    recovered = [row for row in matched if row[2] is not None]
    ratio = len(recovered) / len(matched)

    if ratio < min_match_ratio:
        examples = ", ".join(name for name, _, group in matched if group is None)
        raise GroupingError(
            f"Pattern {pattern!r} recovered a group for only {ratio:.1%} of "
            f"{len(matched)} files (need {min_match_ratio:.0%}). "
            f"Unmatched examples: {examples[:200]}. "
            "Pass --patient-regex with a pattern that matches this dataset's "
            "filenames, or --group-by image to explicitly accept a leaky split."
        )

    return [Record(path=name, label=label, group=group) for name, label, group in recovered]


def group_counts(records):
    """Number of distinct groups, and images per group."""
    per_group = Counter(record.group for record in records)
    return per_group


def split_by_group(records, test_size=0.2, val_size=0.1, seed=42):
    """Partition records into train/val/test with no group spanning two splits.

    The split is *stratified* as well as grouped. In this dataset a patient
    carries a single diagnosis, so every group is class-pure — which means
    distributing groups by overall size alone silently wrecks the class
    balance, and a rare class can vanish from a split entirely. Each class is
    therefore distributed across the splits independently: within a class,
    groups are assigned largest-first to whichever split is furthest below its
    target share. Whole groups stay indivisible and no solver is needed.
    """
    if not 0 < test_size < 1 or not 0 <= val_size < 1:
        raise ValueError("test_size must be in (0, 1) and val_size in [0, 1)")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must leave room for a training set")

    by_group = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)

    # A group's stratum is its majority label. For class-pure groups that is
    # exact; the majority rule only matters for mixed groups, where it keeps
    # one ambiguous group from unbalancing the rest.
    strata = defaultdict(list)
    for group, items in by_group.items():
        label = Counter(record.label for record in items).most_common(1)[0][0]
        strata[label].append((group, items))

    targets = {
        "train": 1.0 - test_size - val_size,
        "val": val_size,
        "test": test_size,
    }
    splits = {name: [] for name in targets}

    for label in sorted(strata):
        # Deterministic order: larger groups first, ties broken by group id, so
        # the same inputs always yield the same partition.
        ordered = sorted(strata[label], key=lambda kv: (-len(kv[1]), kv[0]))
        stratum_total = sum(len(items) for _, items in ordered)
        placed = dict.fromkeys(targets, 0)

        for _group, items in ordered:
            deficits = {
                name: targets[name] - (placed[name] / stratum_total)
                for name in targets
                if targets[name] > 0
            }
            chosen = max(deficits, key=lambda name: (deficits[name], name))
            splits[chosen].extend(items)
            placed[chosen] += len(items)

    assert_no_group_leakage(splits)
    return splits


def kfold_by_group(records, folds=4, seed=42):
    """Partition records into k folds, grouped by slide and stratified by class.

    A single train/test split is fragile when the rare classes have only four
    slides between them: whichever slide lands in test decides the headline
    number. Rotating every slide through the test fold turns "recall was 0.00"
    into "recall was 0.00 in each of four folds", which is a claim that
    survives the obvious follow-up question.
    """
    if folds < MIN_FOLDS:
        raise ValueError(f"folds must be at least {MIN_FOLDS}")

    by_group = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)

    if len(by_group) < folds:
        raise ValueError(f"{len(by_group)} groups cannot fill {folds} folds")

    strata = defaultdict(list)
    for group, items in by_group.items():
        label = Counter(record.label for record in items).most_common(1)[0][0]
        strata[label].append((group, items))

    bins = [[] for _ in range(folds)]
    for label in sorted(strata):
        # Largest groups first, so the big ones are placed while there is still
        # room to balance around them.
        ordered = sorted(strata[label], key=lambda kv: (-len(kv[1]), kv[0]))
        placed = [0] * folds
        for _group, items in ordered:
            target = min(range(folds), key=lambda i: (placed[i], i))
            bins[target].extend(items)
            placed[target] += len(items)

    assert_no_group_leakage({f"fold{i}": bin_ for i, bin_ in enumerate(bins)})
    return bins


def assert_no_group_leakage(splits):
    """Hard check that no group identifier appears in more than one split."""
    seen = {}
    for name, records in splits.items():
        for group in {record.group for record in records}:
            if group in seen:
                raise AssertionError(
                    f"Group {group!r} appears in both {seen[group]!r} and {name!r}. "
                    "The split leaks and every metric derived from it is void."
                )
            seen[group] = name
    return True


def describe(splits):
    """Human-readable split summary — print this and actually read it."""
    lines = []
    total_images = sum(len(records) for records in splits.values())
    total_groups = sum(len({r.group for r in records}) for records in splits.values())

    for name in ("train", "val", "test"):
        records = splits.get(name, [])
        if not records:
            continue
        groups = {record.group for record in records}
        share = len(records) / total_images if total_images else 0
        per_class = Counter(record.label for record in records)
        breakdown = ", ".join(f"{label} {count}" for label, count in sorted(per_class.items()))
        lines.append(
            f"{name:<5} {len(records):>5} images ({share:>5.1%})  "
            f"{len(groups):>4} groups   {breakdown}"
        )

    lines.append(f"{'total':<5} {total_images:>5} images            {total_groups:>4} groups")
    return "\n".join(lines)
