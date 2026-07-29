"""The baseline that should be run before any deep model is trusted.

Each image is reduced to six numbers — the per-channel mean and standard
deviation of its RGB values. No spatial information survives: no cells, no
nuclei, no texture, no morphology. A logistic regression on those six numbers
cannot possibly be doing cytology.

If it scores anywhere near the CNN, the CNN is not doing cytology either. It is
separating slides by their staining and illumination signature, which is a
property of when and how the slide was scanned rather than of the patient.

    python baseline.py --data-dir data/

Writes `docs/baseline_metrics.json`. Evaluated with the same slide-grouped
cross-validation as `cross_validate.py`, so the numbers are directly comparable.
"""

import argparse
import json
import pathlib

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import CLASS_NAMES
from splitting import DEFAULT_PATIENT_PATTERN, infer_group

THUMBNAIL = 32
# More than two classes means the multi-class run rather than NILM-vs-abnormal.
BINARY_CLASS_COUNT = 2
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def colour_features(path):
    """Six numbers: per-channel mean and standard deviation."""
    pixels = np.asarray(
        Image.open(path).convert("RGB").resize((THUMBNAIL, THUMBNAIL)),
        dtype=np.float32,
    )
    return np.concatenate([pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))])


def load(data_dir, classes, pattern):
    root = pathlib.Path(data_dir)
    features, labels, groups = [], [], []

    for index, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            features.append(colour_features(path))
            labels.append(index)
            groups.append(infer_group(path.name, pattern))

    return np.array(features), np.array(labels), np.array(groups)


def evaluate(features, labels, groups, classes, folds):
    scores = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    per_class = {name: [] for name in classes}

    for train_index, test_index in GroupKFold(n_splits=folds).split(features, labels, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ).fit(features[train_index], labels[train_index])
        predicted = model.predict(features[test_index])
        truth = labels[test_index]

        scores["accuracy"].append(float((predicted == truth).mean()))
        scores["balanced_accuracy"].append(float(balanced_accuracy_score(truth, predicted)))
        scores["macro_f1"].append(
            float(f1_score(truth, predicted, average="macro", zero_division=0))
        )
        for index, name in enumerate(classes):
            per_class[name].append(float(recall_score(
                truth, predicted, labels=[index], average="macro", zero_division=0,
            )))

    return scores, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--classes", default=",".join(CLASS_NAMES))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--patient-regex", default=DEFAULT_PATIENT_PATTERN)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    features, labels, groups = load(args.data_dir, classes, args.patient_regex)

    print(f"{len(features)} images, {len(set(groups))} slides, "
          f"{features.shape[1]} features (channel mean + std), {args.folds} folds\n")

    scores, per_class = evaluate(features, labels, groups, classes, args.folds)

    print(f"{'metric':<20}{'mean':>8}   per fold")
    for name, values in scores.items():
        print(f"{name:<20}{np.mean(values):>8.3f}   {np.round(values, 2)}")
    print()
    print(f"{'class recall':<20}{'mean':>8}   per fold")
    for name, values in per_class.items():
        print(f"{name:<20}{np.mean(values):>8.3f}   {np.round(values, 2)}")

    out = pathlib.Path(args.docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Logistic regression on per-channel RGB mean and std only. "
                       "No spatial information. Slide-grouped cross-validation.",
        "classes": classes,
        "folds": args.folds,
        "features": int(features.shape[1]),
        "summary": {
            name: {"mean": round(float(np.mean(values)), 4), "per_fold": values}
            for name, values in scores.items()
        },
        "per_class_recall": {
            name: {"mean": round(float(np.mean(values)), 4), "per_fold": values}
            for name, values in per_class.items()
        },
    }
    multiclass = len(classes) > BINARY_CLASS_COUNT
    filename = "baseline_metrics.json" if multiclass else "baseline_metrics_binary.json"
    (out / filename).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out / filename}")


if __name__ == "__main__":
    main()
