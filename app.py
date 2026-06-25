import os
import torch
from flask_cors import CORS
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, send_from_directory, jsonify
from torchvision import transforms, models
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cpu")

class_names = ["HSIL", "LSIL", "NILM", "SCC"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

model.load_state_dict(torch.load("cervical_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return "Invalid Image", None

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        confidence = confidence.item()

    if confidence < 0.90:
        return "Low Confidence", None

    result = class_names[predicted.item()]
    confidence = round(confidence * 100, 2)

    return result, confidence


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    img_path = None

    if request.method == "POST":

        if "file" not in request.files:
            error = "No file uploaded"
            return render_template("index.html", error=error)

        file = request.files["file"]

        if file.filename == "":
            error = "No image selected"
            return render_template("index.html", error=error)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result, conf = predict_image(filepath)

        if result == "Invalid Image":
            error = "Uploaded file is not a valid image"

        elif result == "Low Confidence":
            error = "Cannot determine the result with confidence. Please upload a clearer image."

        else:
            prediction = result
            confidence = conf
            img_path = f"/uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        img_path=img_path
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result, conf = predict_image(filepath)

    if result == "Invalid Image":
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    if result == "Low Confidence":
        return jsonify({"error": "Low confidence, please upload a clearer image"}), 400

    return jsonify({"prediction": result, "confidence": conf})


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)