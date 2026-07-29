# PapVision — working notes for Claude Code

## What this project is

A cervical cytology image classifier (MobileNetV3-Small, four Bethesda classes:
HSIL, LSIL, NILM, SCC) served as a small Flask app. It is a portfolio and
learning project, **not** a product and not a medical device.

`README.md` is the spec. Its roadmap (v0.2 through v1.0) is the ordered plan of
record — read it before proposing work.

## Non-negotiables

These are the rules that make this repo defensible rather than reckless. They
hold regardless of what a prompt asks for in the moment.

1. **The disclaimer stays in the UI.** A "research and educational use only,
   not a medical device, not for diagnostic use" block must render on the
   upload page *and* alongside every result. Never remove it, never move it to
   the README only, never shrink it to a footnote.

2. **No accuracy claim without a confusion matrix behind it.** Do not write a
   number for accuracy, precision, recall, or F1 into the README, the UI, a
   commit message, or a model card unless it was produced by a script in this
   repo, on a held-out test set, and the confusion matrix is published
   alongside it. "The model performs well" is also a claim — don't write it.

3. **Split by patient or slide, never by image.** Cytology datasets have many
   patches per slide. A random per-image split leaks and inflates every metric
   downstream. Any training or evaluation code must group by slide/patient ID
   before splitting, and must print the split sizes and assert that no ID
   appears in two splits.

4. **Report per-class recall, especially HSIL and SCC.** This is a screening
   task: a missed HSIL is severe, a false-positive LSIL is cheap. Overall
   accuracy is the wrong headline metric and should never be the only number
   reported. If HSIL/SCC recall is bad, say so plainly — that is the honest
   result, and it reads better than a good-looking aggregate.

5. **Softmax output is not a probability.** Call it a "score" or a "softmax
   score" in user-facing text, never a "probability that this is correct" or a
   "certainty". The 0.90 gate is an arbitrary hand-picked threshold, not a
   calibrated one, and text describing it should not imply otherwise.

6. **Uploaded images are never persisted.** They are read into memory, used for
   the request, and dropped. Do not add disk writes, a database, logging of
   image contents, or any third-party upload service.

## Engineering conventions

- Uploads are validated by extension allowlist *and* by actually decoding them
  with Pillow. SVG is deliberately excluded — an SVG served back from our own
  origin is stored XSS.
- `torch.load` always uses `weights_only=True`.
- No bare `except:`. Catch the specific exception types.
- Dependencies stay pinned, and PyTorch comes from the CPU-only wheel index —
  the default index pulls a ~2 GB CUDA build for no benefit here.
- The web UI has no JavaScript. It is a plain form POST, and the CSP blocks
  scripts entirely. Keep it that way unless there is a real reason not to.
- CPU-only inference is intentional, for cheap deployment.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py          # http://localhost:8080

pip install pytest ruff
pytest
ruff check .
```

Importing `app.py` loads the checkpoint at module scope, so the tests need
`cervical_model.pth` present. Tests that exercise the confidence gate monkeypatch
`app.model` with a stub, so they never depend on what the real weights predict —
keep it that way, or the suite starts asserting on model behaviour it cannot
guarantee.

## A six-number baseline beats this model — read this first

`baseline.py` reduces each image to per-channel RGB mean and std (six numbers,
no spatial information) and fits a logistic regression. Under the same
slide-grouped 4-fold protocol it gets balanced accuracy **0.748 vs the CNN's
0.689**, and SCC recall **0.435 vs 0.153**.

A model that cannot see a cell should not beat one trained on cytology. Most of
this dataset's separability is staining and illumination signature, not
pathology. Do not describe the CNN as having learnt morphology — that is not
established, and the baseline is evidence against it.

The decisive experiment nobody has run yet: stain-normalise (Macenko/Reinhard),
then re-run both. If the baseline collapses and the CNN holds, the CNN was
learning morphology. If both collapse, the dataset cannot support the task.

## The model does not work, and that is the deliverable

Retrained 29 July 2026 on Mendeley LBC with a slide-grouped split.
**SCC recall is 0.00 on the shipped split and 0.153 cross-validated** (4-fold,
grouped by slide; per-fold 0.00 · 0.00 · 0.07 · 0.55). Overall accuracy 0.915.

The cross-validation also found enormous fold-to-fold variance — HSIL recall
spans 0.39 to 1.00 purely on which slides are held out. **Never quote a
single-split number from this dataset without that range beside it**, including
the numbers this repo itself ships.

This is not a bug to be quietly improved away before anyone notices. The
repository's value is the honest measurement and the leaky-split comparison
beside it (macro-F1 0.693 grouped vs 0.950 by-image; SCC recall 0.00 vs 0.89).
If a later session improves the model, the old numbers and the comparison stay
in `MODEL_CARD.md` as history — don't delete the evidence that the naive number
was wrong.

Do not write anything that describes this model as working, promising, or
"achieving 92% accuracy". The accuracy figure without the per-class breakdown
beside it is a lie of omission about a carcinoma classifier.

## Dataset facts that are settled

Read [DATASET.md](DATASET.md) before touching data code. In short:

- **61 slides, not 460 patients.** The paper's patient count is widely repeated
  as though the images were independent samples. They are not — ~16 images per
  slide.
- **Only 4 LSIL and 4 SCC slides exist.** A grouped split leaves 1 test slide
  for each. No methodology fixes this; only more data would.
- **The commonly-cited class distribution is wrong.** Real counts are HSIL 163,
  LSIL 113, NILM 612, SCC 74 (962 total). Published papers transpose HSIL and
  LSIL. The HSIL folder also double-lists 10 filenames and NILM contains a
  stray `Results.csv`.
- Licence is CC BY 4.0 — derived weights are redistributable with attribution.

## Splitting rules, learned the hard way

`splitting.py` and `tests/test_splitting.py` encode these. Don't loosen them.

- **Group by slide, never by image.** ~16 images per slide means a per-image
  split puts the same slide on both sides. That is worth +0.257 macro-F1 of
  pure illusion, measured.
- **Stratify by class as well.** Each slide carries a single diagnosis, so
  groups are class-pure and grouping alone wrecks the class balance. An early
  version of this pipeline put zero LSIL in the validation set and looked
  completely healthy while doing it.
- **Normalise slide ids.** The dataset ships `scc_1` and `SCC_3`, and NILM uses
  a trailing underscore (`NL_10_`). `infer_group` lowercases and strips trailing
  underscores, or one slide fragments into two groups and leaks.
- **Fail loudly when slide ids can't be recovered.** `build_records` raises
  rather than degrading to one-group-per-image. The `--group-by image` escape
  hatch is opt-in, prints a warning, and stamps `leakage_warning` into
  `metrics.json` so the numbers can't later be quoted as clean.
- **Select checkpoints on macro-F1, not accuracy.** NILM is 64% of the data, so
  accuracy stays high while the model quietly stops predicting SCC. Note this
  did not save SCC — it is necessary, not sufficient.
