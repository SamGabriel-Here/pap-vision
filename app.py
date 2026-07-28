import base64
import io
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from torchvision import models

from preprocessing import CLASS_NAMES, build_eval_transform

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cervical_model.pth"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
PREVIEW_MAX_EDGE = 384
CONFIDENCE_THRESHOLD = 0.90

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "PAPVISION_ALLOWED_ORIGINS", "http://localhost:8080"
    ).split(",")
    if origin.strip()
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/predict": {"origins": ALLOWED_ORIGINS}})

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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.route("/", methods=["GET", "POST"])
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
    except LowConfidence:
        return render_template(
            "index.html",
            error="Cannot determine the result with confidence. Please upload a clearer image.",
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
        "probabilities": breakdown,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
