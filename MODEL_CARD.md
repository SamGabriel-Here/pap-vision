# Model Card — PapVision cervical cytology classifier

**Version:** v0.2 · trained 29 July 2026 · supersedes an earlier orphaned checkpoint of unknown provenance.

## ⚠️ Do not use this model

**Research and educational use only. Not a medical device. Not for diagnostic use.**

This model **essentially cannot detect squamous cell carcinoma**. On the shipped
split it identified none of the 15 held-out carcinoma images, classifying every
one as HSIL. Across 4-fold slide-grouped cross-validation it recovered **7 of
the dataset's 74 carcinoma fields — a pooled recall of 0.09** (per-fold 0.00,
0.00, 0.07, 0.55). It misses roughly ten carcinoma fields in eleven. In a
screening context that is the worst failure mode available.

It is published because the failure is informative, not because the model works.

## Intended use

Demonstrating honest evaluation methodology for medical imaging models —
specifically what patient-grouped splitting does to reported performance. It is
a teaching artefact and a portfolio piece.

## Out of scope

Every clinical use, without exception. Screening, triage, second reads,
prioritising slides for human review, or any use where a human would see and be
influenced by the output. It has never been validated on clinical slides, across
scanners, staining protocols, or populations, and the evaluation below rests on
a handful of slides.

## Architecture

| | |
|---|---|
| Backbone | `torchvision.models.mobilenet_v3_small`, ImageNet-pretrained |
| Head | `classifier[3]` → `Linear(1024, 4)` |
| Input | 224×224 RGB, ImageNet mean/std normalisation |
| Classes | `["HSIL", "LSIL", "NILM", "SCC"]` — this order is baked into the checkpoint |
| Device | CPU at inference |

## Training data

Mendeley LBC (Hussain et al.), CC BY 4.0 — see [DATASET.md](DATASET.md) for a
full audit including three corrections to its commonly-cited description.

962 images from **61 slides**. The published 2048×1536 JPEGs were stored at
512×384 before training; the model resizes to 224×224 regardless, so this loses
nothing and makes training tractable.

## Split

Grouped by slide, stratified by class, via [splitting.py](splitting.py). No
slide appears in more than one split, asserted rather than assumed.

| | Images | Slides | HSIL | LSIL | NILM | SCC |
|---|---:|---:|---:|---:|---:|---:|
| Train | 618 | 35 | 6 | 2 | 25 | 2 |
| Validation | 108 | 9 | 1 | 1 | 6 | 1 |
| Test | 236 | 17 | 3 | 1 | 12 | 1 |

Slide counts, not image counts, are what matter for how much any of these
numbers can be trusted. Two SCC slides to learn from and one to be tested on.

Class imbalance was handled with inverse-frequency loss weighting, and
checkpoints were selected on validation macro-F1 rather than accuracy — NILM is
64% of the data, so accuracy stays high while the model quietly abandons the
rare classes. Selection on macro-F1 did not save SCC.

## Metrics — held-out slides

Overall accuracy is **0.915**, and that number is worse than useless here.

| Class | Precision | Recall | F1 | Test images | Test slides |
|---|---:|---:|---:|---:|---:|
| HSIL | 0.70 | 0.90 | 0.79 | 41 | 3 |
| LSIL | 1.00 | 1.00 | 1.00 | 27 | 1 |
| NILM | 0.97 | 0.99 | 0.98 | 153 | 12 |
| **SCC** | **0.00** | **0.00** | **0.00** | 15 | 1 |

Balanced accuracy **0.724**, macro-F1 **0.693**.

![Confusion matrix](docs/confusion_matrix.png)

## Cross-validation — every slide rotated through the test fold

The metrics above rest on one held-out slide per rare class, so they were
re-measured with 4-fold cross-validation grouped by slide.

Recall below is **pooled**, not averaged: images of that class the model
recovered, divided by how many the dataset contains. The folds partition the
data, so each image is a test image exactly once and pooling is exact. The fold
mean is shown beside it because for SCC the two disagree badly, and the mean is
the flattering one.

| Class | Pooled recall | Recovered | Fold mean | Min | Max | Per-fold |
|---|---:|---:|---:|---:|---:|---|
| HSIL | 0.767 | 125 / 163 | 0.758 | 0.39 | 1.00 | 0.81 · 0.39 · 1.00 · 0.83 |
| LSIL | 0.858 | 97 / 113 | 0.890 | 0.59 | 1.00 | 0.59 · 0.96 · 1.00 · 1.00 |
| NILM | 0.956 | 585 / 612 | 0.956 | 0.84 | 1.00 | 1.00 · 1.00 · 0.84 · 0.99 |
| **SCC** | **0.095** | **7 / 74** | 0.153 | **0.00** | **0.55** | 0.00 · 0.00 · 0.07 · 0.55 |

SCC is where averaging misleads. Its fold support ranges from 11 test images to
27, and the only fold in which it scored above 0.07 was the fold holding the
fewest carcinoma images. Weighting that fold equally with a fold more than twice
its size lifts the headline from 0.095 to 0.153 — a 62% overstatement of the one
number a screening model is judged on. Regenerate the table with
`python metrics_summary.py`.

Accuracy 0.849 (0.808–0.937) · balanced accuracy 0.689 (0.590–0.840) · macro-F1
0.663 (0.581–0.834). Full detail in [docs/cv_metrics.json](docs/cv_metrics.json),
reproducible with `cross_validate.py`.

Two conclusions, and the second matters more:

1. **SCC is not reliably zero.** One fold reached 0.55. The shipped split's 0.00
   is the worst case, not the universal one. Pooled across every carcinoma image
   in the dataset it is 0.095, which is still unusable.
2. **The variance is the real finding.** HSIL recall spans 0.39 to 1.00 purely on
   which slides were held out. With 61 slides and only 4 in each rare class, a
   single split measures the split as much as the model. Any single-split number
   from this dataset — including the one this card leads with — should be read as
   one draw from a wide distribution.

*Protocol note:* cross-validation folds train on train/test only and use the
final epoch, whereas the shipped single split selects a checkpoint on validation
macro-F1. Some of the spread may be checkpoint selection rather than the split
itself. The size of the variance is not in doubt; its attribution is.

## The baseline this model does not beat

Every image reduced to six numbers — per-channel RGB mean and standard deviation.
No spatial information at all. Logistic regression on those six numbers, same
slide-grouped 4-fold protocol:

| Metric | Colour baseline (6 features) | This model |
|---|---:|---:|
| Accuracy | 0.779 | **0.849** |
| Balanced accuracy | **0.748** | 0.689 |
| Macro-F1 | 0.627 | **0.663** |
| **SCC recall** | **0.435** | 0.153 |

Both columns are fold means. `docs/baseline_metrics.json` does not record
per-fold support, so the baseline cannot be pooled the way the table above pools
this model, and pooling one side alone would invent a gap that the measurement
does not support. On the pooled figure this model does worse still — 0.095.

The baseline wins balanced accuracy and takes nearly three times the carcinoma
recall. A model that cannot see a cell should not be competitive here. That it
is means the separability in this dataset is largely **staining and illumination
signature** — an artefact of when and how each slide was scanned — rather than
cytological morphology.

The honest reading is that this checkpoint has not demonstrably learnt cytology.
It has learnt something, and part of that something is available from average
colour alone.

The decisive follow-up would be stain normalisation (Macenko or Reinhard) before
re-running both. If the colour baseline collapses and the CNN holds, the CNN was
learning morphology after all. If both collapse, the dataset cannot support the
task. That experiment has not been run.

Reproduce with `baseline.py`; results in [docs/baseline_metrics.json](docs/baseline_metrics.json).

## Known failure modes

**SCC is barely predicted at all.** All 15 carcinoma images in the shipped test
split were classified as HSIL, and pooled cross-validated recall is 0.095. The
model sees 2–3 SCC slides in training and does not generalise to an unseen one.
This is not a threshold artefact — on the shipped split, SCC is not the argmax
for any test image.

**Four HSIL images were classified as NILM.** High-grade lesions called normal is
the other severe error, and it happens at roughly 10%.

**LSIL's perfect score means nothing.** All 27 LSIL test images come from a
single slide. The model may have learnt that slide's staining rather than
low-grade lesion morphology, and there is no way to tell from this data.

**Softmax scores are not calibrated.** No temperature scaling was fitted. The
number the app displays is a normalised logit, not a probability of correctness.

**No out-of-distribution rejection.** A flat, structureless colour field —
`RGB(128, 224, 96)`, a saturated green — clears the 0.90 confidence gate,
returning `NILM` at 92.42%. This was found by a grid search over solid colours
and verified against the running app, not assumed. Pure random pixel noise is
*not* the concerning case here: 200 seeded trials of uniform random noise
topped out at 37.25% confidence and were correctly declined by the gate. It is
smooth, saturated, low-frequency colour that the confidence gate cannot
distinguish from real tissue, not high-frequency noise.

## What a leaky split would have claimed

The same architecture, hyperparameters, and seed, split by image instead of by
slide — so images from one slide land on both sides of the boundary:

| Metric | Grouped by slide | Split by image | Inflation |
|---|---:|---:|---:|
| Accuracy | 0.915 | 0.975 | +0.060 |
| Balanced accuracy | 0.724 | 0.947 | +0.223 |
| Macro-F1 | 0.693 | 0.950 | +0.257 |
| **SCC recall** | **0.00** | **0.89** | **+0.89** |

The leaky split reports a model that detects carcinoma nine times in ten. The
grouped split shows it detecting none of them here, and 7 in 74 across all folds.
Both numbers come from the same code, minutes apart. This is the single most
important thing in this repository.

Reproduce with `--group-by image`; that flag stamps a leakage warning into the
metrics file so the numbers cannot later be quoted as clean.

## What would actually be needed

More SCC and LSIL slides, from more sources. Four slides per class cannot
support a claim about either. Beyond that: calibration, an out-of-distribution
gate, slide-level aggregation instead of single-field classification, and
prospective validation on clinical material before the word "screening" is
used at all.

## Citation

> Hussain, Elima (2019), "Liquid based cytology pap smear images for multi-class
> diagnosis of cervical cancer", Mendeley Data, V4, doi: 10.17632/zddtpgzv63.4.
> Licensed CC BY 4.0.
