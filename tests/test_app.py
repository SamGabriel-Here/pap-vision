import base64
import io

import pytest
import torch
from PIL import Image
from werkzeug.datastructures import FileStorage

import app as papvision


def make_image_bytes(fmt="PNG", size=(64, 64), color=(200, 150, 180), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(data, filename="smear.png"):
    return FileStorage(stream=io.BytesIO(data), filename=filename)


class StubModel:
    """Stands in for MobileNetV3 so the confidence gate can be tested directly."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, _tensor):
        return torch.tensor([self.logits])


@pytest.fixture
def client():
    papvision.app.config["TESTING"] = True
    return papvision.app.test_client()


@pytest.fixture
def confident(monkeypatch):
    # Softmax over these puts ~0.9999 on index 0, which is "HSIL".
    monkeypatch.setattr(papvision, "model", StubModel([12.0, 0.0, 0.0, 0.0]))


@pytest.fixture
def unsure(monkeypatch):
    # Near-uniform logits: top score lands around 0.32, well under the 0.90 gate.
    monkeypatch.setattr(papvision, "model", StubModel([1.0, 0.9, 0.8, 0.7]))


# --- upload validation ----------------------------------------------------

def test_missing_file_field_is_rejected():
    with pytest.raises(papvision.UploadError, match="No file uploaded"):
        papvision.read_upload(None)


def test_empty_filename_is_rejected():
    with pytest.raises(papvision.UploadError, match="No image selected"):
        papvision.read_upload(upload(make_image_bytes(), filename=""))


def test_svg_is_rejected_by_the_extension_allowlist():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    with pytest.raises(papvision.UploadError, match="Unsupported file type"):
        papvision.read_upload(upload(svg, filename="payload.svg"))


def test_non_image_bytes_wearing_an_image_extension_are_rejected():
    with pytest.raises(papvision.UploadError, match="not a valid image"):
        papvision.read_upload(upload(b"this is not a PNG", filename="fake.png"))


def test_empty_upload_is_rejected():
    with pytest.raises(papvision.UploadError, match="empty"):
        papvision.read_upload(upload(b"", filename="empty.png"))


def test_grayscale_upload_is_converted_to_rgb():
    image = papvision.read_upload(
        upload(make_image_bytes(mode="L", color=128), filename="gray.png")
    )
    assert image.mode == "RGB"


# --- preprocessing --------------------------------------------------------

def test_preprocessing_produces_a_single_224_batch():
    image = papvision.read_upload(upload(make_image_bytes(size=(37, 512))))
    tensor = papvision.transform(image).unsqueeze(0)
    assert tuple(tensor.shape) == (1, 3, 224, 224)


def test_preview_is_an_inline_png_data_uri_not_a_disk_path():
    image = papvision.read_upload(upload(make_image_bytes(size=(2000, 1000))))
    uri = papvision.preview_data_uri(image)
    assert uri.startswith("data:image/png;base64,")

    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert max(decoded.size) <= papvision.PREVIEW_MAX_EDGE


# --- the confidence gate --------------------------------------------------

def test_confident_prediction_returns_class_and_full_breakdown(confident):
    image = papvision.read_upload(upload(make_image_bytes()))
    prediction, confidence, breakdown = papvision.classify(image)

    assert prediction == "HSIL"
    assert confidence > 99.0
    assert set(breakdown) == set(papvision.CLASS_NAMES)
    assert sum(breakdown.values()) == pytest.approx(100.0, abs=0.1)


def test_scores_below_the_threshold_raise_low_confidence(unsure):
    image = papvision.read_upload(upload(make_image_bytes()))
    with pytest.raises(papvision.LowConfidence) as raised:
        papvision.classify(image)

    assert set(raised.value.probabilities) == set(papvision.CLASS_NAMES)


# --- HTTP surface ---------------------------------------------------------

def test_index_shows_the_disclaimer(client):
    body = client.get("/").get_data(as_text=True)
    assert "Not a medical device" in body
    assert "Not for diagnostic use" in body


def test_result_page_repeats_the_disclaimer_next_to_the_prediction(client, confident):
    response = client.post(
        "/", data={"file": (io.BytesIO(make_image_bytes()), "smear.png")}
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "HSIL" in body
    assert "This is not a diagnosis" in body
    assert body.count("Not a medical device") >= 2


def test_predict_returns_prediction_and_probabilities(client, confident):
    response = client.post(
        "/predict", data={"file": (io.BytesIO(make_image_bytes()), "smear.png")}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["prediction"] == "HSIL"
    assert set(payload["probabilities"]) == set(papvision.CLASS_NAMES)


def test_predict_rejects_a_missing_file(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file uploaded"


def test_predict_rejects_an_undecodable_image(client):
    response = client.post(
        "/predict", data={"file": (io.BytesIO(b"nope"), "fake.png")}
    )
    assert response.status_code == 400
    assert "not a valid image" in response.get_json()["error"]


def test_predict_declines_to_answer_below_the_threshold(client, unsure):
    response = client.post(
        "/predict", data={"file": (io.BytesIO(make_image_bytes()), "smear.png")}
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert "Low confidence" in payload["error"]
    assert "prediction" not in payload


def test_oversized_upload_is_refused(client):
    oversized = b"\x00" * (papvision.MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/predict", data={"file": (io.BytesIO(oversized), "huge.png")}
    )
    assert response.status_code == 413


def test_security_headers_are_set(client):
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'none'" in headers["Content-Security-Policy"]


def test_health_endpoint(client):
    payload = client.get("/health").get_json()
    assert payload["status"] == "ok"
    assert payload["classes"] == papvision.CLASS_NAMES
