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

## The checkpoint is orphaned — this is settled, don't re-litigate it

The training script and dataset behind `cervical_model.pth` **no longer exist**
(confirmed by the author, 28 July 2026). What follows, and should not be argued
around in later sessions:

- The current checkpoint can never be audited. Do not write a model card for it,
  do not infer its dataset, do not report any metric about it.
- `CLASS_NAMES` cannot be verified against it. No artifact remains to check the
  order against, so the mislabelling risk is permanent while these weights ship.
- The only route to a defensible model is a full retrain from a documented
  dataset. Characterising or repairing the existing weights is not an option.

## Open questions that gate the roadmap

Don't guess at these — ask.

- **Are patient identifiers actually recoverable from the dataset filenames?**
  Unverified, and it is the one thing that could still block a clean split.
  Answer it by downloading Mendeley LBC and listing a class folder. If the
  answer is no, do not quietly fall back to a per-image split — see below.
- What is the nearest job-application deadline? Per [[portfolio-strategy]] that
  decides whether the retrain happens at all, or whether the time is better
  spent on a backend-systems project.

Dataset for the retrain is Mendeley LBC (Hussain et al.): 963 images, 460
patients, CC BY 4.0, and the only common public set labelled with exactly these
four Bethesda classes. The licence permits redistributing derived weights with
attribution.

## Splitting rules, learned the hard way

`splitting.py` and `tests/test_splitting.py` encode these. Don't loosen them.

- **Group by patient, never by image.** ~2 images per patient means a per-image
  split puts the same patient on both sides for most patients.
- **Stratify by class as well.** Each patient carries a single diagnosis, so
  groups are class-pure and grouping alone wrecks the class balance. An early
  version of this pipeline put zero LSIL in the validation set and looked
  completely healthy while doing it.
- **Fail loudly when patient ids can't be recovered.** `build_records` raises
  rather than degrading to one-group-per-image. The `--group-by image` escape
  hatch is opt-in, prints a warning, and stamps `leakage_warning` into
  `metrics.json` so the numbers can't later be quoted as clean.
- **Select checkpoints on macro-F1, not accuracy.** NILM is 64% of the data, so
  accuracy stays high while the model quietly stops predicting SCC.
