#pip freeze > requirements.txt
#pip install -r requirements.txt 
from flask import Flask, request, render_template, jsonify
import base64, os, random
import tensorflow as tf
import numpy as np
from PIL import Image
import os, io

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict-digit", methods=["POST", "GET"])
def predict_digit():
    image = request.get_json(silent=True)['image'].split(",")[1]
    image_data = base64.urlsafe_b64decode(image)
    image = Image.open(io.BytesIO(image_data))
    # Convert the RGB image to grayscale image
    image = image.convert("L")

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert the image into numpy array
    image = np.array(image)

    # Reshape the image for the model
    image = image.reshape(1, 28, 28) 

    # Normalize the pixel values in image
    image = image / 255.

    # Set the datatype of image as float32
    image = image.astype(np.float32)
    input_data = image.astype(np.float32)
    values = model.predict(input_data)
    value = np.argmax(values)
    response = { 
        "prediction": str(value),
        "confidence": str('')
    }

    return jsonify(response)