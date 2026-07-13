from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gc
from huggingface_hub import hf_hub_download

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_REPO_ID = "keerthi7879/brain-tumor-models"

# Download .tflite models once (cached locally at startup)
custom_cnn_path = hf_hub_download(repo_id=HF_REPO_ID, filename="custom_cnn_model.tflite")
vgg16_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vgg16_model.tflite")
resnet50_path = hf_hub_download(repo_id=HF_REPO_ID, filename="resnet50_model.tflite")


def run_tflite_prediction(model_path, img_array):
    """Loads a TFLite model, runs one prediction, and frees memory."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    del interpreter
    gc.collect()

    return output[0][0]  # single sigmoid value


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Same one-at-a-time approach, but with TFLite interpreters now
    cnn = run_tflite_prediction(custom_cnn_path, img_array)
    vgg = 1 - run_tflite_prediction(vgg16_path, img_array)
    resnet = 1 - run_tflite_prediction(resnet50_path, img_array)

    final_score = (cnn + vgg + resnet) / 3
    label = (
        "Tumor Detected"
        if final_score > 0.5
        else "No Tumor"
    )
    confidence = max(final_score, 1 - final_score)
    return label, round(float(confidence * 100), 2)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    if request.method == "POST":
        file = request.files["file"]
        uploads_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        save_path = os.path.join(uploads_dir, file.filename)
        file.save(save_path)
        result, confidence = predict_image(save_path)
    return render_template("index.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
