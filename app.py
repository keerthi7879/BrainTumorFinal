from flask import Flask, render_template
import tensorflow as tf

app = Flask(__name__)

# Load models from local models folder
model_cnn = tf.keras.models.load_model("models/custom_cnn_model.h5")
model_vgg = tf.keras.models.load_model("models/vgg16_model.h5")
model_resnet = tf.keras.models.load_model("models/resnet50_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
