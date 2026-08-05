import base64
import hashlib
import io
import json
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image, UnidentifiedImageError
from torchvision import models

from preprocessing import CLASS_NAMES, build_eval_transform

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cervical_model.pth"
MODEL_METADATA_PATH = BASE_DIR / "model_metadata.json"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
PREVIEW_MAX_EDGE = 384
CONFIDENCE_THRESHOLD = 0.90

# CLASS_NAMES is alphabetical because that is the order baked into the
# checkpoint and it must not change. For display, severity order reads better:
# normal through to carcinoma, the way the Bethesda system is taught.
SEVERITY_ORDER = ["NILM", "LSIL", "HSIL", "SCC"]

CLASS_MEANINGS = {
    "NILM": "Negative for intraepithelial lesion or malignancy",
    "LSIL": "Low-grade squamous intraepithelial lesion",
    "HSIL": "High-grade squamous intraepithelial lesion",
    "SCC": "Squamous cell carcinoma",
}

# What the model's measured behaviour is for each class, shown in the UI so a
# result is never presented without the evidence about how much to trust it.
# Figures are from 4-fold slide-grouped cross-validation — see MODEL_CARD.md.
CLASS_RELIABILITY = {
    "NILM": ("0.96", "Most reliable class, and the easiest — 64% of the training data."),
    "LSIL": ("0.89", "Looks strong, but every test image came from a single slide."),
    "HSIL": ("0.76", "Swings between 0.39 and 1.00 depending on which slides are held out."),
    "SCC": ("0.15", "Effectively broken. Usually returns HSIL instead, with high confidence."),
}

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "PAPVISION_ALLOWED_ORIGINS", "http://localhost:8080"
    ).split(",")
    if origin.strip()
]

# Model inference is the expensive part of this service and there is no
# batching, so an unthrottled /predict is a trivial CPU exhaustion vector.
# 20 requests per minute per client IP by default; overridable for load
# testing or when a trusted reverse proxy is doing this job instead.
PREDICT_RATE_LIMIT = os.environ.get("PAPVISION_PREDICT_RATE_LIMIT", "20 per minute")

# In-memory storage keeps a separate counter per *process*. The shipped
# Dockerfile runs gunicorn with 2 workers, so under memory:// storage each
# worker enforces its own copy of the same limit independently and the
# effective ceiling under concurrent load can be up to (workers x limit),
# not exactly limit. Point this at a shared backend such as Redis
# (e.g. "redis://redis:6379") for an exact limit across workers/replicas,
# or run with a single worker.
RATE_LIMIT_STORAGE_URI = os.environ.get("PAPVISION_RATE_LIMIT_STORAGE_URI", "memory://")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/predict": {"origins": ALLOWED_ORIGINS}})

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=RATE_LIMIT_STORAGE_URI,
)

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_metadata(metadata_path=MODEL_METADATA_PATH, model_path=MODEL_PATH):
    """Load and verify the sidecar tying the checkpoint to serving code."""
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing model metadata: {metadata_path}") from error

    expected_classes = metadata.get("class_names")
    if expected_classes != CLASS_NAMES:
        raise RuntimeError(
            f"Model metadata class order {expected_classes!r} does not match serving "
            f"class order {CLASS_NAMES!r}"
        )

    expected_sha = metadata.get("sha256")
    actual_sha = file_sha256(model_path)
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"Model checkpoint checksum mismatch for {model_path.name}: "
            f"metadata has {expected_sha}, file has {actual_sha}"
        )

    return metadata


MODEL_METADATA = load_model_metadata()

device = torch.device("cpu")

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

transform = build_eval_transform()


class UploadError(Exception):
    """A rejected upload. The message is safe to show the user."""


class LowConfidence(Exception):
    """Top softmax score fell below the reporting threshold."""

    def __init__(self, probabilities):
        super().__init__("Low confidence")
        self.probabilities = probabilities


def read_upload(file_storage):
    """Validate an uploaded file and return it as an RGB PIL image."""
    if file_storage is None:
        raise UploadError("No file uploaded")
    if not file_storage.filename:
        raise UploadError("No image selected")

    extension = pathlib.Path(file_storage.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file type")

    raw = file_storage.read()
    if not raw:
        raise UploadError("Uploaded file is empty")

    try:
        Image.open(io.BytesIO(raw)).verify()
        # verify() consumes the file object, so reopen for actual decoding.
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise UploadError("Uploaded file is not a valid image") from None


def classify(image):
    """Return (class name, confidence percent, per-class probabilities)."""
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]

    breakdown = {
        name: round(score * 100, 2)
        for name, score in zip(CLASS_NAMES, probabilities.tolist(), strict=True)
    }
    confidence, index = torch.max(probabilities, 0)

    if confidence.item() < CONFIDENCE_THRESHOLD:
        raise LowConfidence(breakdown)

    return CLASS_NAMES[index.item()], round(confidence.item() * 100, 2), breakdown


def preview_data_uri(image):
    """Downscaled copy of the upload, inlined so nothing is written to disk."""
    thumbnail = image.copy()
    thumbnail.thumbnail((PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE))

    buffer = io.BytesIO()
    thumbnail.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


@app.context_processor
def template_globals():
    """Available to every render, so a result can never be shown without the
    class meanings and measured reliability beside it."""
    return {
        "meanings": CLASS_MEANINGS,
        "reliability": CLASS_RELIABILITY,
        "class_names": SEVERITY_ORDER,
        "threshold_percent": int(CONFIDENCE_THRESHOLD * 100),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
    }


@app.after_request
def set_security_headers(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; form-action 'self'; base-uri 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    message = f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
    if request.path == "/predict":
        return jsonify({"error": message}), 413
    return render_template("index.html", error=message), 413


@app.errorhandler(429)
def rate_limited(_error):
    message = "Too many requests. Please wait a moment and try again."
    if request.path == "/predict":
        return jsonify({"error": message}), 429
    return render_template("index.html", error=message), 429


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.route("/", methods=["GET", "POST"])
@limiter.limit(PREDICT_RATE_LIMIT, methods=["POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    try:
        image = read_upload(request.files.get("file"))
    except UploadError as error:
        return render_template("index.html", error=str(error)), 400

    preview = preview_data_uri(image)

    try:
        prediction, confidence, breakdown = classify(image)
    except LowConfidence as low:
        return render_template(
            "index.html",
            error="Cannot determine the result with confidence. Please upload a clearer image.",
            probabilities=low.probabilities,
            img_src=preview,
        ), 200

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=breakdown,
        img_src=preview,
    )


@app.route("/predict", methods=["POST"])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict():
    try:
        image = read_upload(request.files.get("file"))
    except UploadError as error:
        return jsonify({"error": str(error)}), 400

    try:
        prediction, confidence, breakdown = classify(image)
    except LowConfidence as low:
        return jsonify({
            "error": "Low confidence, please upload a clearer image",
            "probabilities": low.probabilities,
        }), 400

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "softmax_score": confidence,
        "probabilities": breakdown,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
