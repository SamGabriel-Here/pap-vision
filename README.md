# PapVision

A cervical cytology image classifier served as a small Flask web app. Upload a Pap smear image, get a predicted Bethesda-system class and a per-class score breakdown, either through a browser page or a JSON endpoint.

**Status: v0.3 — the service is hardened, the model is orphaned.** The inference path works end to end and the web layer has been tightened, tested, and put under CI. The bundled checkpoint, however, was produced by a training script and dataset that no longer exist, so it cannot be audited and its predictions should not be relied on for anything. See [The checkpoint is orphaned](#the-checkpoint-is-orphaned) and [Roadmap](#roadmap).

---

## ⚠️ Not a medical device

**This project is for research and educational purposes only.**

- It is **not** a diagnostic tool, and it is **not** cleared or approved by any regulatory body (FDA, CDSCO, CE, or otherwise).
- It must **not** be used to make, support, or influence any clinical decision.
- Predictions from this model are **not** a substitute for examination by a qualified cytopathologist.
- The model was trained on a public research dataset and has **not** been validated on real-world clinical slides, across scanners, staining protocols, or patient populations.

If you have a health concern, talk to a doctor. Do not upload your own medical images here expecting a meaningful answer.

This notice is also rendered at the top of the web UI and repeated directly beside every prediction.

---

## What it does

1. You upload an image of a cervical cytology (Pap smear) field.
2. The upload is checked against an extension allowlist and then actually decoded, so a file that merely claims to be an image is rejected.
3. The image is resized to 224×224 and normalised with ImageNet statistics.
4. A fine-tuned MobileNetV3-Small produces logits over four classes.
5. Softmax is applied; if the top score falls below a 0.90 threshold, the app declines to name a class rather than guessing.
6. Otherwise the predicted class is returned, along with the score for **all four** classes.

Uploads are held in memory for the lifetime of the request and never written to disk. The preview shown in the UI is a downscaled copy embedded in the page as a data URI.

### Classes

The four labels follow the Bethesda System for reporting cervical cytology:

| Label | Meaning |
|-------|---------|
| `NILM` | Negative for intraepithelial lesion or malignancy |
| `LSIL` | Low-grade squamous intraepithelial lesion |
| `HSIL` | High-grade squamous intraepithelial lesion |
| `SCC`  | Squamous cell carcinoma |

> The class order in `app.py` is alphabetical (`HSIL`, `LSIL`, `NILM`, `SCC`), matching `ImageFolder`'s default. **This cannot be verified.** The training code and dataset that produced `cervical_model.pth` no longer exist, so there is nothing left to check the order against. If the training run used a different order, every prediction is mislabelled while still looking entirely plausible — and there is no way to tell from inside this repo. Resolving this requires retraining; see [Roadmap v0.2](#v02--retrain-from-a-documented-dataset).

---

## Architecture

```
browser / client
      │  multipart image upload
      ▼
Flask app  (app.py)
      │
      ├── "/"         → HTML page, renders prediction inline
      ├── "/predict"  → JSON API
      └── "/health"   → liveness probe
      │
      ▼
read_upload()                    classify()
      │                                │
      ├── extension allowlist          ├── Resize(224) → ToTensor → Normalize
      ├── Pillow verify + decode       ├── MobileNetV3-Small, final Linear → 4
      └── RGB convert                  ├── softmax → per-class breakdown
                                       └── raise LowConfidence if top < 0.90
```

The model is loaded once at import time onto CPU and kept in `eval()` mode. There is no GPU path — this is intentional for cheap deployment, and a single 224×224 forward pass through MobileNetV3-Small is fast enough on CPU.

The web UI contains no JavaScript. It is a plain form POST, and the Content-Security-Policy blocks scripts entirely.

---

## Project structure

```
app.py                Flask routes, model loading, upload validation, inference
preprocessing.py      Transforms and class names shared by training and serving
splitting.py          Leakage-safe, stratified, patient-grouped dataset splitting
inspect_dataset.py    Pre-flight check: can patients be recovered from filenames?
train.py              Fine-tuning pipeline; writes metrics and a confusion matrix
cervical_model.pth    Trained MobileNetV3-Small weights (4-class head)
requirements.txt      Pinned runtime dependencies, CPU-only PyTorch wheels
requirements-train.txt Training-only extras (scikit-learn, matplotlib, numpy)
pyproject.toml        pytest and ruff configuration
CLAUDE.md             Project constraints for AI-assisted sessions
templates/
  index.html          Upload form, disclaimer, result display
tests/
  test_app.py         Upload validation, preprocessing, confidence gate, HTTP surface
  test_splitting.py   Leakage, stratification, and determinism of the split
  test_inspect_dataset.py  Pattern ranking, including degenerate-grouping traps
.github/workflows/
  ci.yml              Lint and test on every push and pull request
```

`preprocessing.py` exists so training and serving cannot drift apart. If the two applied different resizing or normalisation the model would still run and still answer confidently — it would just be quietly wrong, with nothing in the output to indicate it.

---

## Running it locally

```bash
git clone https://github.com/SamGabriel-Here/pap-vision.git
cd pap-vision

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open <http://localhost:8080>.

Run the tests:

```bash
pip install pytest ruff
pytest
ruff check .
```

For a production-ish run:

```bash
gunicorn -w 2 -b 0.0.0.0:8080 app:app
```

Note that each gunicorn worker loads its own copy of the model into memory.

### Retraining

The pipeline is written and tested, but no model has been trained with it yet. It expects one folder per class:

```
data/
  HSIL/  LSIL/  NILM/  SCC/
```

Before training, check whether the filenames carry a patient identifier. This needs only the standard library, so run it before installing anything:

```bash
python inspect_dataset.py --data-dir data/
```

It reports which patterns recover a patient id and prints the exact `train.py` command to use — or tells you no grouping is possible and what to do instead.

```bash
pip install -r requirements-train.txt
python train.py --data-dir data/ --dry-run    # check the split first
python train.py --data-dir data/ --epochs 25
```

The run prints its split summary before training starts, and refuses to continue unless it can recover a patient identifier from at least 95% of filenames. **Read that summary rather than trusting it** — a leaked split does not crash, it just quietly inflates every number that follows.

If the dataset genuinely does not expose patient identifiers, `--group-by image` forces the leaky split through. It prints a warning, and stamps `leakage_warning` into `docs/metrics.json` so the resulting numbers can never be quoted as clean. `--patient-regex` overrides the filename pattern if the default (leading digits) doesn't match.

Outputs are `cervical_model.pth`, `docs/confusion_matrix.png`, and `docs/metrics.json`.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PAPVISION_ALLOWED_ORIGINS` | `http://localhost:8080` | Comma-separated CORS allowlist for `/predict` |

Uploads are capped at 8 MB (`MAX_CONTENT_LENGTH`).

---

## API

### `POST /predict`

**Request** — `multipart/form-data` with a single `file` field. Allowed extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

```bash
curl -F "file=@sample.png" http://localhost:8080/predict
```

**Success — 200**

```json
{
  "prediction": "LSIL",
  "confidence": 94.31,
  "probabilities": { "HSIL": 2.10, "LSIL": 94.31, "NILM": 3.02, "SCC": 0.57 }
}
```

`confidence` and the values in `probabilities` are softmax scores as percentages. They are **not** probabilities that the answer is correct — see [Known Limitations](#known-limitations).

**Errors — 400**

```json
{ "error": "No file uploaded" }
{ "error": "No image selected" }
{ "error": "Unsupported file type" }
{ "error": "Uploaded file is empty" }
{ "error": "Uploaded file is not a valid image" }
{ "error": "Low confidence, please upload a clearer image", "probabilities": { ... } }
```

**Error — 413**

```json
{ "error": "Image is larger than 8 MB" }
```

### `GET /` and `POST /`

Same pipeline, rendering `templates/index.html` with the prediction, the full score breakdown, an inline preview of the upload, or an error message.

### `GET /health`

```json
{ "status": "ok", "classes": ["HSIL", "LSIL", "NILM", "SCC"] }
```

---

## Model

| | |
|---|---|
| Backbone | `torchvision.models.mobilenet_v3_small` |
| Head | `classifier[3]` replaced with `Linear(in_features, 4)` |
| Input | 224×224 RGB, ImageNet mean/std normalisation |
| Device | CPU |
| Checkpoint | `cervical_model.pth` (plain `state_dict`, 244 tensors) |

### The checkpoint is orphaned

The training script, the dataset, and the split used to produce `cervical_model.pth` **no longer exist**. This is not a documentation gap that will be filled in later — the artifacts are gone.

The practical consequence is that this checkpoint can never be audited. Its dataset, licence, split strategy, augmentation, schedule, and held-out metrics are unrecoverable, and its class ordering is unverifiable. No honest model card can be written for it, and no metric reported about it could be trusted.

**Treat every output of the current checkpoint as meaningless.** The only path to a defensible model here is to retrain from a documented dataset, which is what [v0.2](#v02--retrain-from-a-documented-dataset) now describes.

---

## Known limitations

Being explicit about these, because a classifier in this domain that doesn't state its failure modes is worse than no classifier.

**Model and evaluation**

- **No published metrics, and none are recoverable.** There is no confusion matrix, no per-class recall, no held-out accuracy — and because the training artifacts are gone, none can be produced for this checkpoint. Its real performance is unknown to everyone, including its author.
- **Class ordering is unverifiable.** See [The checkpoint is orphaned](#the-checkpoint-is-orphaned). If it is wrong, every output is a plausible-looking mislabel and nothing in this repo would reveal it.
- **Accuracy would be the wrong headline metric anyway.** In a screening context a missed HSIL is a severe error and a false-positive LSIL is a cheap one. Per-class recall matters far more than overall accuracy.
- **Softmax confidence is not calibrated.** Deep classifiers are systematically overconfident. The number shown as "94.31%" is a normalised logit, not a probability that the answer is correct.
- **No out-of-distribution rejection, and this is easy to demonstrate.** Feeding the API a 400×400 image of uniform random RGB noise returns `NILM` at **90.43%** — clearing the 0.90 gate outright. The threshold does not catch inputs that are not cytology at all.
- **Fixed, arbitrary threshold.** 0.90 was chosen by hand, not derived from a validation sweep.
- **Single-image, single-field.** Real cytology screening reviews an entire slide. This looks at one cropped field in isolation.
- **Unknown generalisation.** No evidence it transfers across scanners, magnifications, staining protocols, or populations.

**Engineering**

- No rate limiting on `/predict`, so the endpoint can be trivially saturated.
- The model checkpoint is committed directly to git rather than stored as a release asset or via LFS.
- Each gunicorn worker loads its own copy of the model; there is no shared-memory or batching path.
- No structured request logging or latency metrics.

Fixed in v0.3, listed here because earlier revisions of this README described them as open: uploads no longer touch disk (so there is no filename collision, no stored-XSS vector, and no unbounded disk growth), `torch.load` uses `weights_only=True`, the bare `except:` is gone, CORS is narrowed to a configurable allowlist, dependencies are pinned, and there is a test suite running in CI.

---

## Roadmap

Ordered by what most needs fixing, not by what's most fun.

### v0.2 — Retrain from a documented dataset

The current checkpoint cannot be audited or repaired, only replaced. This release retrains from scratch against a dataset whose provenance is written down.

Candidate dataset: the **Mendeley LBC** liquid-based cytology set (Hussain et al.), which is the only common public dataset labelled with exactly these four Bethesda classes — 963 images at 2048×1536, drawn from 460 patients, distributed NILM 613 / LSIL 163 / HSIL 113 / SCC 74.

Its licence is **CC BY 4.0**, which permits redistributing derived weights with attribution.

- [x] Write `train.py`, splitting by **patient**, not by image. With roughly two images per patient a random per-image split leaks heavily, and every number downstream would be void
- [x] Print split sizes and assert no patient ID appears in more than one split — then read that output rather than trusting a summary of it
- [x] Stratify the split by class as well as grouping it. Patients are class-pure, so grouping alone silently starves the rare classes — an early version of this pipeline put **zero LSIL** in the validation set while looking entirely healthy
- [x] Handle the class imbalance explicitly (NILM is 64% of the data; predicting NILM always scores 64% accuracy) via inverse-frequency class weighting, and select checkpoints on macro-F1 rather than accuracy
- [x] Ship a pre-flight check (`inspect_dataset.py`) that answers the patient-id question the moment the data lands, and names the exact next command either way
- [x] Pick a licence for the code, and check it against the dataset's redistribution terms
- [ ] Download the dataset and record its exact version and DOI
- [ ] **Confirm patient identifiers are actually recoverable from the filenames.** This is unverified and is the one thing that could still block a clean split
- [ ] Run the training, then replace this section's checkboxes with real numbers
- [ ] Report a confusion matrix and per-class precision, recall, and F1 on the held-out test set
- [ ] Report macro-F1 and balanced accuracy alongside raw accuracy
- [ ] State the test-set support per class next to every metric. SCC has only 74 images in total, so its test-set recall will rest on a handful of images and deserves an explicit error bar rather than a confident decimal
- [ ] Replace `cervical_model.pth`, and confirm the class order the new run actually used
- [ ] Add a `MODEL_CARD.md` covering intended use, out-of-scope use, training data, metrics, and known failure modes

### v0.3 — Harden the service ✅

- [x] Extension allowlist plus real decoding on upload; reject anything that isn't a valid raster image
- [x] Stop writing uploads to disk entirely — this supersedes the planned UUID-renaming and cleanup job
- [x] Set `MAX_CONTENT_LENGTH` (8 MB) with a friendly 413 on both the HTML and JSON paths
- [x] Locked-down `Content-Security-Policy`, `X-Content-Type-Options`, and `Referrer-Policy`; this supersedes the planned `Content-Disposition` header, since uploads are no longer served back
- [x] Narrow CORS to a configurable origin allowlist
- [x] `torch.load(..., weights_only=True)`
- [x] Replace the bare `except:` with specific exception handling
- [x] Pin all dependencies; use the CPU-only PyTorch wheel index to cut image size
- [ ] Basic rate limiting on `/predict`

### v0.4 — Show the model's reasoning

- [ ] Grad-CAM overlay returned alongside every prediction
- [ ] Side-by-side view of the input and the attention heatmap in the web UI
- [ ] A short gallery in the README showing the model attending to nuclei rather than staining artifacts or slide edges
- [ ] Document at least three concrete failure cases with images

### v0.5 — Honest confidence

- [ ] Temperature scaling fitted on a validation set
- [ ] Reliability diagram and expected calibration error published
- [ ] Threshold chosen from a precision/recall sweep rather than hardcoded
- [ ] Out-of-distribution detection (max-logit or energy score) as a separate gate from softmax confidence
- [x] Surface a full per-class score breakdown in the API response, not just the top class
- [x] Prominent in-app disclaimer rendered on the result itself, not only in the README

### v0.6 — Deployment and reproducibility

- [ ] Dockerfile and `docker-compose.yml`
- [x] GitHub Actions running lint and tests on push
- [x] Unit tests for preprocessing, the confidence gate, and both error paths
- [ ] Move `cervical_model.pth` to a GitHub Release asset, downloaded on first run
- [ ] Structured request logging with latency and prediction distribution
- [x] `/health` endpoint

### v1.0 — Beyond a single field

- [ ] Whole-slide input with tiling and aggregation into a slide-level result
- [ ] Cell detection and segmentation before classification, rather than classifying the whole field
- [ ] Compare MobileNetV3 against a stronger backbone (EfficientNet, ConvNeXt-Tiny) and report the accuracy/latency tradeoff
- [ ] Test-time augmentation
- [ ] Optional ONNX Runtime path for faster CPU inference

### Deliberately out of scope

- Any claim of clinical validity
- Any deployment intended for real patient care
- Storing uploaded images beyond the lifetime of a request

---

## Contributing

Issues and pull requests are welcome, particularly around evaluation methodology and dataset documentation. Please don't open issues asking for clinical interpretation of an image.

## Acknowledgements

- PyTorch and torchvision for the MobileNetV3 implementation and pretrained weights
- The Bethesda System for cervical cytology terminology

## License

The code in this repository is MIT licensed — see [LICENSE](LICENSE).

Model weights are a separate question from code. The Mendeley LBC dataset is published under **CC BY 4.0**, which permits redistributing derived works including trained weights, provided the dataset is attributed. Any checkpoint trained from it and shipped here must therefore carry that attribution in `MODEL_CARD.md`. The currently bundled `cervical_model.pth` predates this and has unknown provenance, which is one more reason it should be replaced rather than documented.
