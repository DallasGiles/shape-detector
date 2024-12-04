from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model("shape_detection_model.h5")

# Label mapping
label_mapping = {0: "circle", 1: "square", 2: "triangle"}

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Log the request files
    print("Request files:", request.files)

    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400

    file = request.files["image"]
    print("Received file:", file.filename)

    # Read the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess the image
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = np.argmax(model.predict(image), axis=1)[0]
    label = label_mapping[prediction]

    return jsonify({"label": label})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)