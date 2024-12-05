from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow all origins for debugging

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

model = load_model("backend/shape_detection_model.h5")

label_mapping = {0: "circle", 1: "square", 2: "triangle"}

@app.route("/")
def index():
    return jsonify({"message": "Shape Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Just to ensure we see what's received
    print("Request files:", request.files)

    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    file = request.files["image"]
    print("Received file:", file.filename)

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = np.argmax(model.predict(image), axis=1)[0]
    label = label_mapping[prediction]

    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(debug=True)