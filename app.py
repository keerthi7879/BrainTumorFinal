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

# Download models once (files cached locally, not loaded into memory yet)
custom_cnn_path = hf_hub_download(repo_id=HF_REPO_ID, filename="custom_cnn_model.keras")
vgg16_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vgg16_model.keras")
resnet50_path = hf_hub_download(repo_id=HF_REPO_ID, filename="resnet50_model.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Load, predict, unload - one model at a time (saves memory)
    model_cnn = tf.keras.models.load_model(custom_cnn_path)
    cnn = model_cnn.predict(img_array, verbose=0)
    del model_cnn
    gc.collect()

    model_vgg = tf.keras.models.load_model(vgg16_path)
    vgg = 1 - model_vgg.predict(img_array, verbose=0)
    del model_vgg
    gc.collect()

    model_resnet = tf.keras.models.load_model(resnet50_path)
    resnet = 1 - model_resnet.predict(img_array, verbose=0)
    del model_resnet
    gc.collect()

    final_score = (cnn + vgg + resnet) / 3
    label = (
        "Tumor Detected"
        if final_score > 0.5
        else "No Tumor"
    )
    confidence = max(final_score, 1-final_score)
    return label, round(float(confidence*100),2)

@app.route("/", methods=["GET","POST"])
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

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
