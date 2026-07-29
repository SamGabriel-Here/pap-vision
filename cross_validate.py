"""Slide-grouped k-fold cross-validation.

A single train/test split is not enough here. SCC and LSIL have four slides
each, so whichever slide happens to land in the test set decides the reported
per-class recall outright. Rotating every slide through the test fold turns a
number that could be an accident of the split into one that either holds up or
doesn't.

    python cross_validate.py --data-dir data/ --folds 4

Writes `docs/cv_metrics.json` and prints per-fold and aggregate per-class recall.
No checkpoint is saved — this measures the *procedure*, not one model.
"""

import argparse
import json
import pathlib
import random
import statistics

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from preprocessing import CLASS_NAMES
from splitting import DEFAULT_PATIENT_PATTERN, build_records, kfold_by_group
from train import (
    build_loaders,
    build_model,
    class_weights,
    discover,
    fit,
    pick_device,
    predict_all,
)


def evaluate_fold(train_records, test_records, args, device):
    loaders = build_loaders(
        {"train": train_records, "test": test_records}, args.data_dir, args.batch_size
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_records, device))
    model = fit(build_model(device), loaders, device, criterion, args)

    true, pred = predict_all(model, loaders["test"], device)
    report = classification_report(
        true, pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": round(report["accuracy"], 4),
        "balanced_accuracy": round(balanced_accuracy_score(true, pred), 4),
        "macro_f1": round(f1_score(true, pred, average="macro", zero_division=0), 4),
        "per_class": {
            name: {
                "recall": round(report[name]["recall"], 4),
                "precision": round(report[name]["precision"], 4),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
    }


def summarise(fold_results):
    """Mean and range per class. With four folds a range is more honest than
    a standard deviation pretending to be a confidence interval."""
    summary = {}
    for name in CLASS_NAMES:
        recalls = [f["per_class"][name]["recall"] for f in fold_results]
        support = [f["per_class"][name]["support"] for f in fold_results]
        summary[name] = {
            "recall_mean": round(statistics.fmean(recalls), 4),
            "recall_min": min(recalls),
            "recall_max": max(recalls),
            "per_fold_recall": recalls,
            "test_images_per_fold": support,
        }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = [f[metric] for f in fold_results]
        summary[metric] = {
            "mean": round(statistics.fmean(values), 4),
            "min": min(values),
            "max": max(values),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patient-regex", default=DEFAULT_PATIENT_PATTERN)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    records = build_records(discover(args.data_dir), pattern=args.patient_regex)
    bins = kfold_by_group(records, folds=args.folds, seed=args.seed)

    print(f"{len(records)} images, {len({r.group for r in records})} slides, "
          f"{args.folds} folds, training on {device}\n")
    for index, bin_ in enumerate(bins):
        slides = {r.group for r in bin_}
        print(f"  fold {index}: {len(bin_):>4} images, {len(slides):>2} slides")
    print()

    fold_results = []
    for index in range(args.folds):
        test_records = bins[index]
        train_records = [r for j, b in enumerate(bins) if j != index for r in b]
        print(f"--- fold {index + 1}/{args.folds} "
              f"({len(train_records)} train / {len(test_records)} test) ---")

        result = evaluate_fold(train_records, test_records, args, device)
        fold_results.append(result)

        recalls = "  ".join(
            f"{name} {result['per_class'][name]['recall']:.2f}" for name in CLASS_NAMES
        )
        print(f"  macro-F1 {result['macro_f1']:.4f}   recall: {recalls}\n")

    summary = summarise(fold_results)

    print("=" * 62)
    print(f"{'class':<8}{'mean':>8}{'min':>8}{'max':>8}   per-fold recall")
    for name in CLASS_NAMES:
        stats = summary[name]
        per_fold = " ".join(f"{value:.2f}" for value in stats["per_fold_recall"])
        print(f"{name:<8}{stats['recall_mean']:>8.3f}{stats['recall_min']:>8.2f}"
              f"{stats['recall_max']:>8.2f}   {per_fold}")
    print()
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        stats = summary[metric]
        print(f"{metric:<20} {stats['mean']:.4f}  (range {stats['min']}–{stats['max']})")

    out = pathlib.Path(args.docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv_metrics.json").write_text(json.dumps({
        "folds": args.folds,
        "grouped_by": "slide",
        "epochs": args.epochs,
        "seed": args.seed,
        "per_fold": fold_results,
        "summary": summary,
    }, indent=2) + "\n")
    print(f"\nwrote {out / 'cv_metrics.json'}")


if __name__ == "__main__":
    main()
