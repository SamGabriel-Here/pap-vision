# Model Card — PapVision cervical cytology classifier

**Version:** v0.2 · trained 29 July 2026 · supersedes an earlier orphaned checkpoint of unknown provenance.

## ⚠️ Do not use this model

**Research and educational use only. Not a medical device. Not for diagnostic use.**

This model has **zero recall on squamous cell carcinoma** against a held-out slide.
It identified none of the 15 carcinoma images in the test set, classifying every
one as HSIL. In a screening context that is the worst failure mode available.
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

## Known failure modes

**SCC is never predicted.** All 15 carcinoma images were classified as HSIL. The
model was trained on 2 SCC slides and did not generalise to a third. This is not
a threshold artefact — SCC is not the argmax for any test image.

**Four HSIL images were classified as NILM.** High-grade lesions called normal is
the other severe error, and it happens at roughly 10%.

**LSIL's perfect score means nothing.** All 27 LSIL test images come from a
single slide. The model may have learnt that slide's staining rather than
low-grade lesion morphology, and there is no way to tell from this data.

**Softmax scores are not calibrated.** No temperature scaling was fitted. The
number the app displays is a normalised logit, not a probability of correctness.

**No out-of-distribution rejection.** Uniform random noise still maps into a
pathology class with high confidence.

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
grouped split shows it never detects carcinoma at all. Both numbers come from
the same code, minutes apart. This is the single most important thing in this
repository.

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
