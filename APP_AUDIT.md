# PapVision app audit

Date: 2026-08-05

## What the app does

PapVision is a small Flask application that serves a four-class cervical cytology image classifier. The user uploads a single field image through the HTML form at `/` or the JSON API at `/predict`. The server validates the upload, decodes it with Pillow, converts it to RGB, preprocesses it with the same 224x224 ImageNet-normalized transform used at evaluation time, and runs a MobileNetV3-Small checkpoint over the classes `HSIL`, `LSIL`, `NILM`, and `SCC`.

The app returns the top class only when the top softmax score is at least 0.90. Otherwise, the JSON API returns a low-confidence error with the score breakdown. The browser UI embeds a downscaled preview as a data URI and does not write uploads to disk.

## Validated workflow

Commands run in a scratch virtual environment:

```bash
python -m pip install -r requirements-train.txt pytest ruff
python -m pytest -q
python -m ruff check .
```

Results:

- `72 passed`
- `ruff check .` passed

Note: `requirements.txt` is sufficient for serving, but the full test suite imports `baseline.py`, so test environments need `requirements-train.txt` or at least `scikit-learn` in addition to serving dependencies.

## Strengths

- The README, model card, and UI are unusually transparent about model failure and non-clinical use.
- Upload handling is safer than a typical prototype: allowlisted extensions, real image decoding, size limit, RGB conversion, and no disk persistence.
- Training and serving share preprocessing through `preprocessing.py`, reducing train/serve skew.
- Class order is centralized and tested, which matters because the checkpoint depends on it.
- Dataset splitting is grouped by slide/patient and asserts no leakage, avoiding the main evaluation trap in this dataset.
- Tests cover upload validation, confidence gating, security headers, class ordering, splitting, inspection utilities, and the baseline.

## Implemented from this audit

- Model checkpoint/class-order integrity guard (`model_metadata.json` + startup validation in `app.py`), with tests for the success path, class-order drift, and checksum drift.
- Low-confidence HTML results now show the same per-class softmax breakdown the JSON API already returned, instead of a bare error message.
- `/predict` now also returns `softmax_score` alongside the existing `confidence` field (kept for backward compatibility), and the README documents both.
- Added `Dockerfile` and `.dockerignore`: `requirements.txt`-only image, non-root `papvision` user, `gunicorn` with 2 workers, `HEALTHCHECK` against `/health`. Actually built and ran locally: `docker build` succeeded (1.36 GB image), the container started healthy, `curl /health` returned `200`, and `curl -F file=@... /predict` returned real inference JSON from inside the container. CI now has a second job that builds the image and smoke-tests `/health` on every push/PR.

## Issues and risks

1. **The shipped model is not useful for clinical interpretation.** Cross-validated SCC recall is about 0.15, and the shipped split has 0.00 SCC recall. This is acknowledged clearly, but it remains the central app limitation.
2. **The 0.90 softmax gate does not protect against confident errors.** The model can call SCC images HSIL with high confidence, so the gate catches hesitation rather than correctness.
3. **Softmax scores are uncalibrated.** The UI describes them as not probabilities of correctness. The API now also exposes `softmax_score` alongside `confidence` to reduce the misleading name, but no calibration (temperature scaling, ECE) has been fitted.
4. **There is no out-of-distribution rejection.** Non-cytology or synthetic images may still be mapped to one of the four disease classes with high softmax.
5. **The dataset is too small at the slide level.** Rare classes have only a few slides, so per-class metrics have high fold-to-fold variance.
6. ~~Low-confidence HTML results hide the score breakdown.~~ Fixed: the HTML page now renders the same breakdown the JSON API always returned.
7. ~~Deployment hardening was partly incomplete.~~ Fixed: `Dockerfile`, `.dockerignore`, a non-root runtime user, a container `HEALTHCHECK`, and a CI job that builds and smoke-tests the image now exist.
8. **Full-test dependency expectations are easy to miss.** Running tests after only `requirements.txt` fails because `scikit-learn` is intentionally training-only but needed by baseline tests.

## Recommended improvements

### Highest impact (not yet done — requires new experiments/data, out of scope for a code review pass)

- Treat the current model as a teaching artifact only. Do not improve the UI in ways that make the prediction feel more authoritative.
- Add calibration work: temperature scaling on validation folds, reliability diagrams, expected calibration error.
- Add an OOD/quality gate before classification. Start simple with image-quality heuristics and an energy/entropy threshold, then validate on non-cytology negatives.
- Run stain normalization experiments, such as Macenko or Reinhard, and compare the CNN against the six-colour baseline again. This directly tests whether the CNN learned morphology or staining/illumination artifacts.

### Engineering (remaining)

None identified that are both low-risk and independently verifiable without new user-specified requirements. Note: checksum enforcement in CI is already effectively covered — `app.py` validates the checkpoint against `model_metadata.json` at import time, so `pytest` in CI already fails hard on any mismatch (verified locally by corrupting the recorded hash and re-running the suite, which failed at collection with `RuntimeError: Model checkpoint checksum mismatch`, then restoring it and confirming `72 passed` again).

### Research

- Increase slide-level diversity before expecting real model gains.
- Report patient/slide counts beside every metric, not just image counts.
- Evaluate binary `NILM` vs `abnormal` separately from four-class grading if the four-class dataset remains too sparse.
- Consider multiple instance learning or cell-level segmentation only after obtaining enough slide-level diversity to make validation meaningful.
