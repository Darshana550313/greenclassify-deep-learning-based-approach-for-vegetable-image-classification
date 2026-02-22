import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

# Load model
model = load_model("vegetable_classifier_model.h5")


class_names = sorted(os.listdir("Vegetable Images/train"))


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Home Page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction Page
@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return render_template("prediction.html", error="Please upload image")

       
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        
        img = load_img(filepath, target_size=(150,150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

       
        result = model.predict(img_array)
        predicted_index = np.argmax(result)
        predicted_class = class_names[predicted_index]
        confidence = round(result[0][predicted_index] * 100, 2)

        return render_template(
            "prediction.html",
            pred=predicted_class,
            confidence=confidence,
            image_path=filepath
        )

    return render_template("prediction.html")

@app.route("/logout")
def logout():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
