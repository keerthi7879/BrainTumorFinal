
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

BASE_DIR="/content/drive/MyDrive/BrainTumorFinal"

model_cnn=tf.keras.models.load_model(
f"{BASE_DIR}/models/custom_cnn_model.h5"
)

model_vgg=tf.keras.models.load_model(
f"{BASE_DIR}/models/vgg16_model.h5"
)

model_resnet=tf.keras.models.load_model(
f"{BASE_DIR}/models/resnet50_model.h5"
)


def predict_image(img_path):

    img=image.load_img(
        img_path,
        target_size=(224,224)
    )

    img_array=image.img_to_array(img)

    img_array=np.expand_dims(
        img_array,
        axis=0
    )

    img_array=img_array/255.0


    cnn=model_cnn.predict(
        img_array,
        verbose=0
    )[0][0]

    vgg=1-model_vgg.predict(
        img_array,
        verbose=0
    )[0][0]

    resnet=1-model_resnet.predict(
        img_array,
        verbose=0
    )[0][0]

    final_score=(cnn+vgg+resnet)/3

    label=(
        "Tumor Detected"
        if final_score>0.5
        else "No Tumor"
    )

    confidence=max(
        final_score,
        1-final_score
    )

    return label, round(confidence*100,2)


@app.route("/",methods=["GET","POST"])
def home():

    result=None
    confidence=None

    if request.method=="POST":

        file=request.files["file"]

        save_path=os.path.join(
            BASE_DIR,
            "uploads",
            file.filename
        )

        file.save(save_path)

        result,confidence=(
            predict_image(save_path)
        )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
