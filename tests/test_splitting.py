"""Tests for the leakage-safe split.

These matter more than the rest of the suite combined. A split bug does not
crash and does not look wrong — it just inflates every metric downstream. So
the properties are asserted directly rather than eyeballed from a summary.
"""

import math

import pytest

from splitting import (
    GroupingError,
    Record,
    assert_no_group_leakage,
    build_records,
    describe,
    group_counts,
    infer_group,
    split_by_group,
)

# Mirrors Mendeley LBC's shape: 963 images across ~2 images per patient, with
# the real class distribution (NILM 613 / LSIL 163 / HSIL 113 / SCC 74).
CLASS_SIZES = {"NILM": 613, "LSIL": 163, "HSIL": 113, "SCC": 74}
IMAGES_PER_PATIENT = 2
EXPECTED_PATIENTS = sum(math.ceil(n / IMAGES_PER_PATIENT) for n in CLASS_SIZES.values())


def synthetic_files():
    """(filename, label) pairs, ~2 images per patient.

    Crucially each patient is class-pure: a patient carries one diagnosis, so
    all of their images share a label. Modelling this wrongly hides exactly the
    stratification bug these tests exist to catch.
    """
    files, patient = [], 0
    for label, count in CLASS_SIZES.items():
        remaining = count
        while remaining:
            for _ in range(min(IMAGES_PER_PATIENT, remaining)):
                files.append((f"{patient:04d}_{len(files):04d}.jpg", label))
            remaining -= min(IMAGES_PER_PATIENT, remaining)
            patient += 1
    return files


# --- group inference ------------------------------------------------------

def test_leading_digits_are_read_as_the_group():
    assert infer_group("0142_a.jpg") == "0142"


def test_unmatched_filenames_yield_no_group():
    assert infer_group("patient-a.jpg") is None


def test_a_pattern_that_mostly_fails_raises_rather_than_degrading():
    files = [("slideA.jpg", "NILM"), ("slideB.jpg", "LSIL"), ("0001.jpg", "SCC")]
    with pytest.raises(GroupingError, match="recovered a group for only"):
        build_records(files)


def test_the_grouping_error_names_the_escape_hatches():
    with pytest.raises(GroupingError) as raised:
        build_records([("a.jpg", "NILM")])
    message = str(raised.value)
    assert "--patient-regex" in message
    assert "--group-by image" in message


def test_a_custom_pattern_can_rescue_a_different_naming_scheme():
    files = [("case-77-img3.jpg", "NILM"), ("case-77-img4.jpg", "NILM")]
    records = build_records(files, pattern=r"case-(\d+)")
    assert {record.group for record in records} == {"77"}


def test_empty_input_is_rejected():
    with pytest.raises(GroupingError, match="No input files"):
        build_records([])


# --- the leakage property -------------------------------------------------

def test_no_patient_appears_in_two_splits():
    records = build_records(synthetic_files())
    splits = split_by_group(records)

    train = {r.group for r in splits["train"]}
    val = {r.group for r in splits["val"]}
    test = {r.group for r in splits["test"]}

    assert train & test == set()
    assert train & val == set()
    assert val & test == set()


def test_every_image_lands_in_exactly_one_split():
    records = build_records(synthetic_files())
    splits = split_by_group(records)

    placed = [r.path for split in splits.values() for r in split]
    assert len(placed) == len(records)
    assert len(set(placed)) == len(records)


def test_a_deliberately_leaked_split_is_caught():
    shared = Record(path="0001_a.jpg", label="HSIL", group="0001")
    splits = {
        "train": [shared],
        "val": [],
        "test": [Record(path="0001_b.jpg", label="HSIL", group="0001")],
    }
    with pytest.raises(AssertionError, match="appears in both"):
        assert_no_group_leakage(splits)


def test_split_sizes_land_near_the_requested_ratios():
    records = build_records(synthetic_files())
    splits = split_by_group(records, test_size=0.2, val_size=0.1)
    total = len(records)

    assert len(splits["test"]) / total == pytest.approx(0.2, abs=0.03)
    assert len(splits["val"]) / total == pytest.approx(0.1, abs=0.03)
    assert len(splits["train"]) / total == pytest.approx(0.7, abs=0.03)


def test_every_class_appears_in_every_split():
    """Patients carry one diagnosis each, so groups are class-pure. Distributing
    them by size alone drops rare classes out of a split entirely."""
    splits = split_by_group(build_records(synthetic_files()))

    for name in ("train", "val", "test"):
        present = {record.label for record in splits[name]}
        assert present == set(CLASS_SIZES), f"{name} split is missing {set(CLASS_SIZES) - present}"


@pytest.mark.parametrize("label", sorted(CLASS_SIZES))
def test_each_class_is_split_near_the_requested_ratios(label):
    splits = split_by_group(build_records(synthetic_files()), test_size=0.2, val_size=0.1)
    counts = {
        name: sum(1 for record in splits[name] if record.label == label)
        for name in ("train", "val", "test")
    }
    total = sum(counts.values())

    assert counts["train"] / total == pytest.approx(0.7, abs=0.08)
    assert counts["val"] / total == pytest.approx(0.1, abs=0.08)
    assert counts["test"] / total == pytest.approx(0.2, abs=0.08)


def test_scc_the_rarest_class_still_reaches_the_test_set():
    splits = split_by_group(build_records(synthetic_files()))
    scc_test = sum(1 for record in splits["test"] if record.label == "SCC")
    assert scc_test > 0


def test_the_split_is_deterministic():
    records = build_records(synthetic_files())
    first = split_by_group(records)
    second = split_by_group(records)

    assert [r.path for r in first["test"]] == [r.path for r in second["test"]]


def test_groups_stay_intact_when_one_patient_dominates():
    files = [(f"0001_{i}.jpg", "NILM") for i in range(50)]
    files += [(f"{p:04d}_0.jpg", "LSIL") for p in range(2, 60)]
    splits = split_by_group(build_records(files))

    holders = [name for name, records in splits.items()
               if any(r.group == "0001" for r in records)]
    assert len(holders) == 1


def test_impossible_ratios_are_rejected():
    records = build_records(synthetic_files())
    with pytest.raises(ValueError, match="leave room for a training set"):
        split_by_group(records, test_size=0.8, val_size=0.3)


# --- reporting ------------------------------------------------------------

def test_group_counts_reflect_multiple_images_per_patient():
    counts = group_counts(build_records(synthetic_files()))
    assert len(counts) == EXPECTED_PATIENTS
    assert sum(counts.values()) == sum(CLASS_SIZES.values())
    assert max(counts.values()) == IMAGES_PER_PATIENT


def test_a_mixed_group_is_stratified_by_its_majority_label():
    files = [("0001_a.jpg", "NILM"), ("0001_b.jpg", "NILM"), ("0001_c.jpg", "SCC")]
    files += [(f"{p:04d}_0.jpg", "NILM") for p in range(2, 40)]
    splits = split_by_group(build_records(files))

    holders = [name for name, records in splits.items()
               if any(record.group == "0001" for record in records)]
    assert len(holders) == 1


def test_describe_reports_images_groups_and_class_breakdown():
    summary = describe(split_by_group(build_records(synthetic_files())))
    assert "train" in summary and "test" in summary
    assert "groups" in summary
    assert "NILM" in summary
    assert "963 images" in summary
