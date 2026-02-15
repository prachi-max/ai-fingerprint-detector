from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

# ---------------- CONFIG ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224

# Load trained CNN model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_fingerprint_spoof.h5")

print("Looking for model at:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ---------------- FUNCTIONS ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = "uploaded_fingerprint.png"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            prob = model.predict(img)[0][0]

            confidence = round(prob * 100, 2)

            if prob > 0.5:
                prediction = "LIVE FINGERPRINT"
            else:
                prediction = "SPOOF FINGERPRINT"

    return render_template(
        "home.html",
        prediction=prediction,
        confidence=confidence,
        image=image_path
    )

# ---------------- RUN ----------------
# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
