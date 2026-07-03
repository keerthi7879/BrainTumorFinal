from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safe model loading
model_cnn = None
model_vgg = None
model_resnet = None

cnn_path = os.path.join(BASE_DIR, "models", "custom_cnn_model.h5")
vgg_path = os.path.join(BASE_DIR, "models", "vgg16_model.h5")
resnet_path = os.path.join(BASE_DIR, "models", "resnet50_model.h5")

if os.path.exists(cnn_path):
    model_cnn = tf.keras.models.load_model(cnn_path)

if os.path.exists(vgg_path):
    model_vgg = tf.keras.models.load_model(vgg_path)

if os.path.exists(resnet_path):
    model_resnet = tf.keras.models.load_model(resnet_path)


def predict_image(img_path):

    if model_cnn is None or model_vgg is None or model_resnet is None:
        return "Models not loaded", 0

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    cnn = model_cnn.predict(img_array, verbose=0)
    vgg = 1 - model_vgg.predict(img_array, verbose=0)
    resnet = 1 - model_resnet.predict(img_array, verbose=0)

    final_score = (cnn + vgg + resnet)/3

    label = "Tumor Detected" if final_score > 0.5 else "No Tumor"
    confidence = max(final_score, 1-final_score)

    return label, round(float(confidence*100),2)


@app.route("/", methods=["GET","POST"])
def home():

    result = None
    confidence = None

    if request.method=="POST":

        file = request.files["file"]

        uploads_dir = os.path.join(BASE_DIR,"uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        save_path = os.path.join(
            uploads_dir,
            file.filename
        )

        file.save(save_path)

        result, confidence = predict_image(save_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
