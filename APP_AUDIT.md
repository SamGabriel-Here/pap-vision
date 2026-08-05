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

- `68 passed`
- `ruff check .` passed

Note: `requirements.txt` is sufficient for serving, but the full test suite imports `baseline.py`, so test environments need `requirements-train.txt` or at least `scikit-learn` in addition to serving dependencies.

## Strengths

- The README, model card, and UI are unusually transparent about model failure and non-clinical use.
- Upload handling is safer than a typical prototype: allowlisted extensions, real image decoding, size limit, RGB conversion, and no disk persistence.
- Training and serving share preprocessing through `preprocessing.py`, reducing train/serve skew.
- Class order is centralized and tested, which matters because the checkpoint depends on it.
- Dataset splitting is grouped by slide/patient and asserts no leakage, avoiding the main evaluation trap in this dataset.
- Tests cover upload validation, confidence gating, security headers, class ordering, splitting, inspection utilities, and the baseline.

## Issues and risks

1. **The shipped model is not useful for clinical interpretation.** Cross-validated SCC recall is about 0.15, and the shipped split has 0.00 SCC recall. This is acknowledged clearly, but it remains the central app limitation.
2. **The 0.90 softmax gate does not protect against confident errors.** The model can call SCC images HSIL with high confidence, so the gate catches hesitation rather than correctness.
3. **Softmax scores are uncalibrated.** The UI describes them as not probabilities of correctness, but the API field is still named `confidence`, which can invite overinterpretation by client developers.
4. **There is no out-of-distribution rejection.** Non-cytology or synthetic images may still be mapped to one of the four disease classes with high softmax.
5. **The dataset is too small at the slide level.** Rare classes have only a few slides, so per-class metrics have high fold-to-fold variance.
6. **Low-confidence HTML results hide the score breakdown.** The JSON API returns probabilities on low confidence, but the browser user only sees a generic failure message and preview.
7. **Deployment hardening is partly incomplete.** CI is already present, but there is no Dockerfile, production deployment example, model checksum, or model metadata file binding checkpoint, class order, architecture, and preprocessing.
8. **Full-test dependency expectations are easy to miss.** Running tests after only `requirements.txt` fails because `scikit-learn` is intentionally training-only but needed by baseline tests.

## Recommended improvements

### Highest impact

- Treat the current model as a teaching artifact only. Do not improve the UI in ways that make the prediction feel more authoritative.
- Add calibration work: temperature scaling on validation folds, reliability diagrams, expected calibration error, and a clearer API field name such as `softmax_score`.
- Add an OOD/quality gate before classification. Start simple with image-quality heuristics and an energy/entropy threshold, then validate on non-cytology negatives.
- Run stain normalization experiments, such as Macenko or Reinhard, and compare the CNN against the six-colour baseline again. This directly tests whether the CNN learned morphology or staining/illumination artifacts.

### Engineering

- Keep CI as the enforcement point for `requirements-train.txt`, `ruff check .`, and `pytest`, and consider adding model checksum validation there if the artifact is updated.
- Add a model metadata sidecar containing architecture, class order, preprocessing constants, training commit, dataset version, metric summary, and SHA256 of the checkpoint.
- Make the JSON API return `softmax_score` while keeping `confidence` temporarily for backward compatibility.
- Show the low-confidence score breakdown in the browser UI, clearly labelled as uncalibrated softmax.
- Add Docker or a minimal deployment guide with `gunicorn`, `PAPVISION_ALLOWED_ORIGINS`, max upload size, and health check.

### Research

- Increase slide-level diversity before expecting real model gains.
- Report patient/slide counts beside every metric, not just image counts.
- Evaluate binary `NILM` vs `abnormal` separately from four-class grading if the four-class dataset remains too sparse.
- Consider multiple instance learning or cell-level segmentation only after obtaining enough slide-level diversity to make validation meaningful.
