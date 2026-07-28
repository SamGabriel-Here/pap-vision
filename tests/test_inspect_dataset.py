"""Tests for the dataset inspector.

The trap this guards against: a pattern that matches every filename but assigns
each image its own group. That looks like success — 100% matched — while
grouping nothing, so the resulting split leaks exactly as badly as a random one.
"""

from inspect_dataset import evaluate_pattern, rank_patterns


def test_a_pattern_that_groups_is_recognised():
    names = ["0001_a.jpg", "0001_b.jpg", "0002_a.jpg", "0002_b.jpg"]
    stats = evaluate_pattern(names, r"^(\d+)")

    assert stats["match_ratio"] == 1.0
    assert stats["groups"] == 2
    assert stats["max_per_group"] == 2
    assert stats["groups_anything"] is True


def test_a_pattern_matching_everything_but_grouping_nothing_is_rejected():
    """100% match ratio is not evidence of grouping."""
    names = ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg"]
    stats = evaluate_pattern(names, r"^(\d+)")

    assert stats["match_ratio"] == 1.0
    assert stats["groups"] == len(names)
    assert stats["groups_anything"] is False


def test_a_pattern_that_matches_nothing_scores_zero():
    stats = evaluate_pattern(["alpha.jpg", "beta.jpg"], r"^(\d+)")

    assert stats["match_ratio"] == 0.0
    assert stats["groups"] == 0
    assert stats["groups_anything"] is False


def test_ranking_recommends_a_real_grouping():
    names = [f"patient_{p:03d}_{i}.jpg" for p in range(40) for i in range(2)]
    usable, _ = rank_patterns(names)

    assert usable, "expected at least one usable pattern"
    assert usable[0]["groups"] == 40
    assert usable[0]["max_per_group"] == 2


def test_ranking_finds_nothing_when_filenames_are_bare_serials():
    names = [f"{i:04d}.jpg" for i in range(200)]
    usable, scored = rank_patterns(names)

    assert usable == []
    assert any(stats["match_ratio"] == 1.0 for stats in scored), (
        "a pattern should still match; it just must not count as usable"
    )


def test_partial_matches_below_the_threshold_are_not_usable():
    names = [f"{p:03d}_{i}.jpg" for p in range(10) for i in range(2)]
    names += ["unlabelled_scan.jpg"] * 5
    usable, _ = rank_patterns(names, min_match_ratio=0.95)

    assert all(stats["match_ratio"] >= 0.95 for stats in usable)


def test_a_pattern_that_collapses_everything_into_one_group_is_rejected():
    """'Everything before the first underscore' captures "patient" for all of
    these — one group, no split possible, and it would win a fewest-groups
    ranking outright."""
    names = [f"patient_{p:03d}_{i}.jpg" for p in range(40) for i in range(2)]
    stats = evaluate_pattern(names, r"^([^_]+)_")

    assert stats["match_ratio"] == 1.0
    assert stats["groups"] == 1

    usable, scored = rank_patterns(names)
    collapsing = next(s for s in scored if s["pattern"] == r"^([^_]+)_")
    assert collapsing["usable"] is False
    assert all(s["groups"] > 1 for s in usable)


def test_hyphen_separated_names_are_grouped():
    names = ["case-12-a.jpg", "case-12-b.jpg", "case-13-a.jpg", "case-13-b.jpg"]
    usable, _ = rank_patterns(names)

    assert usable
    assert usable[0]["groups"] == 2
