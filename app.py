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

custom_cnn_path = hf_hub_download(repo_id=HF_REPO_ID, filename="custom_cnn_model.tflite")
vgg16_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vgg16_model.tflite")
resnet50_path = hf_hub_download(repo_id=HF_REPO_ID, filename="resnet50_model.tflite")


def run_tflite_prediction(model_path, img_array):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    del interpreter
    gc.collect()

    return output[0][0]


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    cnn_raw = run_tflite_prediction(custom_cnn_path, img_array)
    vgg_raw = run_tflite_prediction(vgg16_path, img_array)
    resnet_raw = run_tflite_prediction(resnet50_path, img_array)

    cnn = cnn_raw
    vgg = 1 - vgg_raw
    resnet = 1 - resnet_raw

    final_score = (cnn + vgg + resnet) / 3
    label = "Tumor Detected" if final_score > 0.5 else "No Tumor"
    confidence = max(final_score, 1 - final_score)

    # Individual model results (each model's own confidence %)
    model_results = {
        "Custom CNN": round(float(max(cnn, 1 - cnn) * 100), 2),
        "VGG16": round(float(max(vgg, 1 - vgg) * 100), 2),
        "ResNet50": round(float(max(resnet, 1 - resnet) * 100), 2),
    }

    return label, round(float(confidence * 100), 2), model_results


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    model_results = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files["file"]
        uploads_dir = os.path.join(BASE_DIR, "static", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        save_path = os.path.join(uploads_dir, file.filename)
        file.save(save_path)

        result, confidence, model_results = predict_image(save_path)
        uploaded_image = f"uploads/{file.filename}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        model_results=model_results,
        uploaded_image=uploaded_image,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



