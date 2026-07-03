from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Google Drive BASE_DIR configuration
BASE_DIR = "/content/drive/MyDrive/BrainTumorFinal"

# Models disabled temporarily to prevent GitHub deployment crash
model_cnn = None
model_vgg = None
model_resnet = None


def predict_image(img_path):
    # Temporary safe return statement to prevent system crash
    return "Model Not Available", 0.0


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None

    if request.method == "POST":

        file = request.files["file"]

        # Automatically creating upload directory if missing
        os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

        save_path = os.path.join(
            BASE_DIR,
            "uploads",
            file.filename
        )

        file.save(save_path)

        result, confidence = (
            predict_image(save_path)
        )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

