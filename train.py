"""Fine-tune MobileNetV3-Small on a four-class cervical cytology dataset.

Expects a directory of class folders:

    data/
      HSIL/  LSIL/  NILM/  SCC/

Splitting is by patient, not by image — see `splitting.py` for why that is the
whole ballgame on this dataset. The run prints its split summary and refuses to
continue if patient identifiers cannot be recovered from the filenames.

    python train.py --data-dir data/ --epochs 25

Outputs `cervical_model.pth`, `docs/confusion_matrix.png`, and
`docs/metrics.json`.
"""

import argparse
import json
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from preprocessing import CLASS_NAMES, build_eval_transform, build_train_transform
from splitting import (
    DEFAULT_PATIENT_PATTERN,
    Record,
    build_records,
    describe,
    group_counts,
    split_by_group,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class CytologyDataset(Dataset):
    def __init__(self, records, root, transform, classes=CLASS_NAMES):
        self.records = records
        self.root = pathlib.Path(root)
        self.transform = transform
        self.classes = list(classes)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.root / record.label / record.path).convert("RGB")
        return self.transform(image), self.classes.index(record.label)


def discover(data_dir, classes=CLASS_NAMES):
    """Collect (filename, label) pairs from one folder per class."""
    root = pathlib.Path(data_dir)
    missing = [name for name in classes if not (root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"Missing class folders in {root}: {', '.join(missing)}. "
            f"Expected one directory per class: {', '.join(classes)}"
        )

    files = []
    for label in classes:
        for path in sorted((root / label).iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                files.append((path.name, label))

    if not files:
        raise SystemExit(f"No images found under {root}")
    return files


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(records, device, classes=CLASS_NAMES):
    """Inverse-frequency weights. NILM is ~64% of this dataset, so a model that
    always answers NILM scores 64% accuracy while being clinically useless."""
    counts = torch.tensor(
        [sum(1 for r in records if r.label == name) for name in classes],
        dtype=torch.float,
    )
    weights = counts.sum() / (len(classes) * counts.clamp(min=1))
    return weights.to(device)


def build_model(device, classes=CLASS_NAMES):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    return model.to(device)


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_images, batch_targets in loader:
            images = batch_images.to(device)
            targets = batch_targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += targets.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def predict_all(model, loader, device):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            pred.extend(outputs.argmax(1).cpu().tolist())
            true.extend(targets.tolist())
    return np.array(true), np.array(pred)


def write_report(true, pred, out_dir, grouped_by, classes=CLASS_NAMES):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(classes)))
    matrix = confusion_matrix(true, pred, labels=labels)
    report = classification_report(
        true, pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )

    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(cmap="Blues", colorbar=False)
    display.figure_.suptitle(f"PapVision — held-out test set (split by {grouped_by})")
    display.figure_.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    metrics = {
        "split_grouped_by": grouped_by,
        "leakage_warning": None if grouped_by == "patient" else (
            "Split was by image, not by patient. Images from the same patient "
            "appear in both train and test, so every metric here is inflated "
            "and should be read as an optimistic upper bound, not a result."
        ),
        "balanced_accuracy": round(balanced_accuracy_score(true, pred), 4),
        "macro_f1": round(f1_score(true, pred, average="macro", zero_division=0), 4),
        "accuracy": round(report["accuracy"], 4),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in classes
        },
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": matrix.tolist()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nHeld-out test set")
    print(classification_report(
        true, pred, labels=labels, target_names=classes, zero_division=0,
    ))
    print(f"balanced accuracy {metrics['balanced_accuracy']}  "
          f"macro-F1 {metrics['macro_f1']}")

    for name in [n for n in ("HSIL", "SCC", "ABNORMAL") if n in classes]:
        stats = metrics["per_class"][name]
        print(f"  {name} recall {stats['recall']} on {stats['support']} test images "
              f"— a missed {name} is the expensive error here")

    return metrics


def fit(model, loaders, device, criterion, args):
    """Train, tracking the best validation macro-F1 rather than accuracy.

    Accuracy is dominated by NILM here, so a model that quietly stops
    predicting SCC would still look like it was improving.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], device, criterion, optimizer)
        scheduler.step()

        line = f"epoch {epoch:>3}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.3f}"

        if "val" in loaders:
            true, pred = predict_all(model, loaders["val"], device)
            score = f1_score(true, pred, average="macro", zero_division=0)
            line += f"   val macro-F1 {score:.4f}"
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                line += "  *"
        print(line)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nrestored best checkpoint (val macro-F1 {best_score:.4f})")

    return model


def build_loaders(splits, data_dir, batch_size, classes=CLASS_NAMES):
    return {
        name: DataLoader(
            CytologyDataset(
                records, data_dir,
                build_train_transform() if name == "train" else build_eval_transform(),
                classes=classes,
            ),
            batch_size=batch_size,
            shuffle=(name == "train"),
        )
        for name, records in splits.items() if records
    }


def resolve_records(files, args):
    """Attach group identifiers, or fail loudly rather than split leakily."""
    if args.group_by == "image":
        print("\n!! --group-by image: each image is its own group. Patients will "
              "span train and test, and every metric produced by this run is "
              "inflated. Only acceptable if patient ids genuinely are not "
              "recoverable, and it must be said out loud in the README.\n")
        return [Record(path=name, label=label, group=name) for name, label in files]
    return build_records(files, pattern=args.patient_regex)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="cervical_model.pth")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--patient-regex", default=DEFAULT_PATIENT_PATTERN,
                        help="Regex capturing the patient/slide id from a filename")
    parser.add_argument("--group-by", choices=["patient", "image"], default="patient",
                        help="'image' explicitly accepts a leaky split; the "
                             "resulting metrics are stamped as invalid")
    parser.add_argument("--classes", default=",".join(CLASS_NAMES),
                        help="Comma-separated class folders. Override to train a "
                             "coarser task, e.g. 'NILM,ABNORMAL'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the split and exit without training")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    records = resolve_records(discover(args.data_dir, classes), args)

    counts = group_counts(records)
    print(f"{len(records)} images across {len(counts)} groups "
          f"(max {max(counts.values())} images in one group)")

    splits = split_by_group(records, test_size=args.test_size,
                            val_size=args.val_size, seed=args.seed)
    print(describe(splits))
    print("no group appears in more than one split — verified\n")

    if args.dry_run:
        print("dry run — stopping before training")
        return

    device = pick_device(args.device)
    print(f"training on {device}")

    loaders = build_loaders(splits, args.data_dir, args.batch_size, classes)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(splits["train"], device, classes))
    model = fit(build_model(device, classes), loaders, device, criterion, args)

    true, pred = predict_all(model, loaders["test"], device)
    write_report(true, pred, pathlib.Path(args.docs_dir), args.group_by, classes)

    torch.save(model.state_dict(), args.out)
    print(f"\nwrote {args.out} and {args.docs_dir}/")
    print(f"class order baked into this checkpoint: {classes}")


if __name__ == "__main__":
    main()
