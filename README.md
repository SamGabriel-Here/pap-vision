# PapVision

A cervical cytology image classifier served as a small Flask web app, and an
honest account of why it doesn't work.

**Status: v0.2 — retrained, evaluated, and documented.** The model is now
reproducible from `train.py` against a cited dataset, with a published confusion
matrix and per-class metrics. Those metrics say the model has **zero recall on
squamous cell carcinoma**. That result is the point of this repository.

---

## ⚠️ Not a medical device

**This project is for research and educational purposes only.**

- It is **not** a diagnostic tool, and it is **not** cleared or approved by any regulatory body (FDA, CDSCO, CE, or otherwise).
- It must **not** be used to make, support, or influence any clinical decision.
- Predictions are **not** a substitute for examination by a qualified cytopathologist.
- It has **not** been validated on real-world clinical slides, across scanners, staining protocols, or patient populations.

If you have a health concern, talk to a doctor. Do not upload your own medical images here expecting a meaningful answer.

This notice is also rendered at the top of the web UI and repeated beside every prediction.

---

## The headline result

The same architecture, the same hyperparameters, the same seed. The only
difference is how the train/test split was drawn.

| Metric | Split by **slide** | Split by **image** | Inflation |
|---|---:|---:|---:|
| Accuracy | 0.915 | 0.975 | +0.060 |
| Balanced accuracy | 0.724 | 0.947 | +0.223 |
| Macro-F1 | 0.693 | 0.950 | +0.257 |
| **SCC recall** | **0.00** | **0.89** | **+0.89** |

Mendeley LBC contains 962 images taken from only **61 slides** — roughly 16
images per slide. Split those images randomly and the same slide lands on both
sides of the boundary, so the model only has to recognise a slide's staining and
illumination signature rather than its pathology.

Do that, and you get a model that appears to detect carcinoma nine times out of
ten. Split by slide instead, and it detects carcinoma **zero** times out of
fifteen. Both numbers came from the same code, minutes apart.

Published work on this dataset routinely reports accuracy in the high nineties.

---

## What the honest model actually does

Overall accuracy is 0.915, which tells you almost nothing.

| Class | Precision | Recall | F1 | Test images | Test slides |
|---|---:|---:|---:|---:|---:|
| HSIL | 0.70 | 0.90 | 0.79 | 41 | 3 |
| LSIL | 1.00 | 1.00 | 1.00 | 27 | 1 |
| NILM | 0.97 | 0.99 | 0.98 | 153 | 12 |
| **SCC** | **0.00** | **0.00** | **0.00** | 15 | 1 |

![Confusion matrix](docs/confusion_matrix.png)

Every one of the 15 carcinoma images was classified as HSIL. Not one was
predicted SCC. Four HSIL images were called NILM — high-grade lesions read as
normal, the other severe error in a screening context.

Feeding a held-out SCC image to the running app returns **`HSIL` at 90.67%
confidence**, clearing the 0.90 threshold that is supposed to make it decline.
The gate does not catch this, because the model is not uncertain — it is
confidently wrong.

LSIL's perfect score is not good news either: all 27 LSIL test images come from
a single slide, so the model may have learnt that slide rather than low-grade
lesion morphology. Nothing in this dataset can distinguish those.

Full detail in [MODEL_CARD.md](MODEL_CARD.md); dataset audit in [DATASET.md](DATASET.md).

---

## What it does

1. You upload an image of a cervical cytology (Pap smear) field.
2. The upload is checked against an extension allowlist and then actually decoded, so a file that merely claims to be an image is rejected.
3. The image is resized to 224×224 and normalised with ImageNet statistics.
4. A fine-tuned MobileNetV3-Small produces logits over four classes.
5. Softmax is applied; if the top score falls below 0.90, the app declines to name a class.
6. Otherwise the predicted class is returned, along with the score for **all four** classes.

Uploads are held in memory for the lifetime of the request and never written to disk. The preview in the UI is a downscaled copy embedded as a data URI.

### Classes

| Label | Meaning |
|-------|---------|
| `NILM` | Negative for intraepithelial lesion or malignancy |
| `LSIL` | Low-grade squamous intraepithelial lesion |
| `HSIL` | High-grade squamous intraepithelial lesion |
| `SCC`  | Squamous cell carcinoma |

The class order baked into the checkpoint is `["HSIL", "LSIL", "NILM", "SCC"]`,
set by `preprocessing.CLASS_NAMES` and used by `train.py`, so serving and
training cannot disagree about it.

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

The model loads once at import onto CPU in `eval()` mode. There is no GPU path at inference — intentional, for cheap deployment. The web UI contains no JavaScript; it is a plain form POST and the CSP blocks scripts entirely.

---

## Project structure

```
app.py                Flask routes, model loading, upload validation, inference
preprocessing.py      Transforms and class names shared by training and serving
splitting.py          Leakage-safe, stratified, slide-grouped dataset splitting
inspect_dataset.py    Pre-flight check: can slides be recovered from filenames?
train.py              Fine-tuning pipeline; writes metrics and a confusion matrix
cervical_model.pth    Trained weights (v0.2, reproducible from train.py)
DATASET.md            Dataset audit, including corrections to its published description
MODEL_CARD.md         Intended use, metrics, failure modes
requirements.txt      Pinned runtime dependencies, CPU-only PyTorch wheels
requirements-train.txt Training-only extras (scikit-learn, matplotlib, numpy)
docs/
  confusion_matrix.png            slide-grouped split — the honest one
  confusion_matrix_leaky_split.png  by-image split, for comparison
  metrics.json, metrics_leaky_split.json
templates/index.html  Upload form, disclaimer, result display
tests/                52 tests: upload handling, preprocessing, splitting, pattern ranking
.github/workflows/ci.yml   Lint and test on every push
```

`preprocessing.py` exists so training and serving cannot drift. If the two applied different resizing or normalisation the model would still run and still answer confidently — it would just be quietly wrong.

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

```bash
pip install pytest ruff
pytest && ruff check .
```

For a production-ish run: `gunicorn -w 2 -b 0.0.0.0:8080 app:app` — each worker loads its own copy of the model.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PAPVISION_ALLOWED_ORIGINS` | `http://localhost:8080` | Comma-separated CORS allowlist for `/predict` |

Uploads are capped at 8 MB (`MAX_CONTENT_LENGTH`).

---

## Reproducing the model

Download [Mendeley LBC](https://data.mendeley.com/datasets/zddtpgzv63/4) (CC BY 4.0, 2.23 GB) into one folder per class named `HSIL/ LSIL/ NILM/ SCC/`. The published folders use long descriptive names and the filenames look like `NL_10_ (5).jpg` — see [DATASET.md](DATASET.md).

Check the slide structure before training. This needs only the standard library:

```bash
python inspect_dataset.py --data-dir data/
```

It reports which filename patterns recover a slide id and prints the exact `train.py` command — or tells you no grouping is possible and what to do instead.

```bash
pip install -r requirements-train.txt
python train.py --data-dir data/ --dry-run          # read the split first
python train.py --data-dir data/ --test-size 0.25 --val-size 0.10 --epochs 25
```

Training refuses to start unless it recovers a slide id from at least 95% of filenames. **Read the split summary rather than trusting it** — a leaked split does not crash, it quietly inflates everything downstream.

To reproduce the leaky comparison, add `--group-by image`. That flag prints a warning and stamps `leakage_warning` into `docs/metrics.json`, so those numbers can never be quoted as clean.

Outputs: `cervical_model.pth`, `docs/confusion_matrix.png`, `docs/metrics.json`.

---

## API

### `POST /predict`

`multipart/form-data` with a single `file` field. Allowed: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

```bash
curl -F "file=@sample.png" http://localhost:8080/predict
```

**200**

```json
{
  "prediction": "HSIL",
  "confidence": 90.67,
  "probabilities": { "HSIL": 90.67, "LSIL": 0.09, "NILM": 7.23, "SCC": 2.01 }
}
```

`confidence` and the values in `probabilities` are softmax scores as percentages. They are **not** probabilities that the answer is correct. The example above is a real response to a held-out **SCC** image.

**400** — `No file uploaded`, `No image selected`, `Unsupported file type`, `Uploaded file is empty`, `Uploaded file is not a valid image`, `Low confidence, please upload a clearer image` (with `probabilities`).

**413** — `Image is larger than 8 MB`.

### `GET /health`

```json
{ "status": "ok", "classes": ["HSIL", "LSIL", "NILM", "SCC"] }
```

---

## Known limitations

**Model and evaluation**

- **No SCC recall at all.** Zero of 15 held-out carcinoma images identified. This is a total failure on the class that matters most.
- **Trained on 2 SCC slides and 2 LSIL slides.** Four slides exist per class in the entire dataset. No methodology fixes a sample that small; the LSIL and SCC numbers are anecdotes with error bars spanning most of the unit interval.
- **61 slides, not 460 patients.** The paper's patient count is widely repeated as though the images were independent. They are not. See [DATASET.md](DATASET.md).
- **Softmax confidence is not calibrated.** No temperature scaling. A held-out SCC image returns HSIL at 90.67%.
- **The 0.90 gate does not catch the failures.** It catches uncertainty, and this model is confidently wrong rather than uncertain.
- **No out-of-distribution rejection.** Uniform random noise maps into a pathology class with high confidence.
- **Single-field.** Real screening reviews a whole slide; this classifies one cropped field.
- **Unknown generalisation.** One institution, one microscope, one staining protocol.

**Engineering**

- No rate limiting on `/predict`.
- The checkpoint is committed to git rather than stored as a release asset.
- Each gunicorn worker loads its own copy of the model; no batching.
- No structured request logging or latency metrics.

---

## Roadmap

### v0.2 — Retrain from a documented dataset ✅

- [x] Document the dataset, its source, its licence, and its real class distribution
- [x] Split by **slide**, not by image, and stratify by class
- [x] Assert no slide appears in more than one split
- [x] Handle class imbalance via inverse-frequency weighting; select on macro-F1
- [x] Publish a confusion matrix and per-class precision, recall, F1
- [x] Report macro-F1 and balanced accuracy alongside raw accuracy
- [x] Quantify what a leaky split would have claimed instead
- [x] `MODEL_CARD.md`

### v0.3 — Harden the service ✅

- [x] Extension allowlist plus real decoding; uploads never touch disk
- [x] `MAX_CONTENT_LENGTH` with a friendly 413 on both paths
- [x] Locked-down CSP, `X-Content-Type-Options`, `Referrer-Policy`
- [x] CORS narrowed to a configurable allowlist
- [x] `torch.load(..., weights_only=True)`; no bare `except:`
- [x] Pinned dependencies on the CPU-only PyTorch wheel index
- [ ] Basic rate limiting on `/predict`

### v0.4 — Show the model's reasoning

- [ ] Grad-CAM overlay alongside every prediction — start with the SCC images it calls HSIL
- [ ] Side-by-side input and heatmap in the web UI
- [ ] README gallery of the model attending to nuclei vs staining artifacts

### v0.5 — Honest confidence

- [ ] Temperature scaling fitted on the validation slides
- [ ] Reliability diagram and expected calibration error
- [ ] Threshold from a precision/recall sweep rather than hardcoded
- [ ] Out-of-distribution detection as a separate gate from softmax confidence
- [x] Full per-class score breakdown in the API response
- [x] Prominent in-app disclaimer on the result itself

### v0.6 — Deployment and reproducibility

- [ ] Dockerfile and `docker-compose.yml`
- [x] GitHub Actions running lint and tests on push
- [x] Unit tests for preprocessing, the confidence gate, and both error paths
- [ ] Move `cervical_model.pth` to a release asset
- [x] `/health` endpoint

### v1.0 — Beyond a single field

- [ ] **More SCC and LSIL slides from other sources** — the binding constraint
- [ ] Grouped k-fold cross-validation instead of one split, given how few slides there are
- [ ] Whole-slide input with tiling and slide-level aggregation
- [ ] Compare against a stronger backbone and report the accuracy/latency tradeoff

### Deliberately out of scope

- Any claim of clinical validity
- Any deployment intended for real patient care
- Storing uploaded images beyond the lifetime of a request

---

## Contributing

Issues and pull requests welcome, particularly around evaluation methodology and dataset documentation. Please don't open issues asking for clinical interpretation of an image.

## Acknowledgements

- Hussain, Elima (2019), "Liquid based cytology pap smear images for multi-class diagnosis of cervical cancer", Mendeley Data, V4, [doi:10.17632/zddtpgzv63.4](https://doi.org/10.17632/zddtpgzv63.4), CC BY 4.0
- PyTorch and torchvision for MobileNetV3 and its pretrained weights
- The Bethesda System for cervical cytology terminology

## License

Code is MIT licensed — see [LICENSE](LICENSE). The training data is CC BY 4.0, which permits redistributing derived weights with attribution; that attribution is in [MODEL_CARD.md](MODEL_CARD.md).
