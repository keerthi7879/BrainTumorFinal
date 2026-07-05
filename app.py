from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Create models folder automatically
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
# Google Drive model IDs
model_files = {
    os.path.join(models_dir, "custom_cnn_model.h5"):
    "1inTYEAQ2jI-8tuezSUYmv3Q5lhxd5FoZ",
    os.path.join(models_dir, "resnet50_model.h5"):
    "1C5C1tOsfUtnHf4VHuxt63TtONum0Y0IN",
    os.path.join(models_dir, "vgg16_model.h5"):
    "1Gw_zEDrym7-Q4_WJYPiiQ1Sn_MjwPa5Y",
}
# Download models if missing
for path, file_id in model_files.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False, fuzzy=True)
# Load models
model_cnn = tf.keras.models.load_model(
    os.path.join(models_dir, "custom_cnn_model.h5")
)
model_vgg = tf.keras.models.load_model(
    os.path.join(models_dir, "vgg16_model.h5")
)
model_resnet = tf.keras.models.load_model(
    os.path.join(models_dir, "resnet50_model.h5")
)
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    cnn = model_cnn.predict(img_array, verbose=0)
    vgg = 1 - model_vgg.predict(
        img_array,
        verbose=0
    )
    resnet = 1 - model_resnet.predict(
        img_array,
        verbose=0
    )
    final_score = (cnn + vgg + resnet) / 3
    label = (
        "Tumor Detected"
        if final_score > 0.5
        else "No Tumor"
    )
    confidence = max(
        final_score,
        1-final_score
    )
    return label, round(float(confidence*100),2)
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    confidence = None
    if request.method == "POST":
        file = request.files["file"]
        uploads_dir = os.path.join(
            BASE_DIR,
            "uploads"
        )
        os.makedirs(
            uploads_dir,
            exist_ok=True
        )
        save_path = os.path.join(
            uploads_dir,
            file.filename
        )
        file.save(save_path)
        result, confidence = predict_image(
            save_path
        )
    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )
if __name__=="__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
