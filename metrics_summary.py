"""Per-class recall, pooled across cross-validation folds.

The interface used to hard-code `SCC: 0.15`, the unweighted mean of the four
fold recalls. That number flatters the model. The folds do not carry equal
weight — SCC test support ranges from 11 images to 27 — and the only fold in
which SCC scored above 0.07 was the fold with the fewest SCC images in it.
Averaging the fold recalls therefore lets the smallest fold set the headline.

Pooling is well defined here because the four folds partition the dataset:
every image is a test image exactly once, so pooled recall is simply the
number of images of a class the model recovered divided by how many exist.
Pooled, SCC recall is 7/74 = 0.09, not 0.15.

Deriving these figures rather than typing them is the point. A hand-copied
number drifts the moment the metrics are regenerated, and this repository's
whole claim is that its numbers are measured rather than asserted.
"""

import json
import pathlib
from dataclasses import dataclass

BASE_DIR = pathlib.Path(__file__).resolve().parent
CV_METRICS_PATH = BASE_DIR / "docs" / "cv_metrics.json"

# cv_metrics.json stores recalls rounded to four decimal places, so
# recall * support reconstructs the underlying integer count to well within
# half a unit. Anything further out means the file is not what we think it is.
_COUNT_TOLERANCE = 0.01


@dataclass(frozen=True)
class ClassRecall:
    """Pooled recall for one class, with the fold-to-fold spread beside it."""

    name: str
    recovered: int
    support: int
    per_fold: tuple

    @property
    def recall(self):
        return self.recovered / self.support

    @property
    def minimum(self):
        return min(self.per_fold)

    @property
    def maximum(self):
        return max(self.per_fold)

    @property
    def fold_mean(self):
        """The figure this module exists to replace. Kept so the two can be
        compared in tests and in the model card, not so it can be quoted."""
        return sum(self.per_fold) / len(self.per_fold)


def load_cv_metrics(path=CV_METRICS_PATH):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing cross-validation metrics: {path}") from error


def class_recall_summary(cv_metrics=None):
    """Pooled per-class recall derived from the published fold results."""
    cv_metrics = cv_metrics if cv_metrics is not None else load_cv_metrics()
    folds = cv_metrics["per_fold"]
    if not folds:
        raise RuntimeError("Cross-validation metrics contain no folds")

    names = sorted(folds[0]["per_class"])
    summary = {}

    for name in names:
        recovered = 0
        support = 0
        per_fold = []

        for fold in folds:
            entry = fold["per_class"][name]
            exact = entry["recall"] * entry["support"]
            rounded = round(exact)
            if abs(exact - rounded) > _COUNT_TOLERANCE:
                raise RuntimeError(
                    f"Cannot recover an integer count for {name}: "
                    f"recall {entry['recall']} x support {entry['support']} = {exact}"
                )
            recovered += rounded
            support += entry["support"]
            per_fold.append(entry["recall"])

        if support == 0:
            raise RuntimeError(f"No test images for {name} in any fold")

        summary[name] = ClassRecall(
            name=name,
            recovered=recovered,
            support=support,
            per_fold=tuple(per_fold),
        )

    return summary


def main():
    summary = class_recall_summary()
    header = f"{'class':<6}{'pooled':>9}{'fold mean':>11}{'min':>7}{'max':>7}{'recovered':>12}"
    print(header)
    print("-" * len(header))
    for name in sorted(summary, key=lambda n: summary[n].recall, reverse=True):
        stats = summary[name]
        print(
            f"{name:<6}{stats.recall:>9.4f}{stats.fold_mean:>11.4f}"
            f"{stats.minimum:>7.2f}{stats.maximum:>7.2f}"
            f"{stats.recovered:>7} / {stats.support:<4}"
        )
    print(
        "\nPooled recall is the share of that class's images the model recovered "
        "across\nall folds. The fold mean weights a 11-image fold the same as a "
        "27-image one."
    )


if __name__ == "__main__":
    main()
