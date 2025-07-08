#!/usr/bin/env python

from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import base64
import urllib

from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load model once during startup
MODEL_PATH = 'save.h5'
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Routes
@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    try:
        file = request.files['imagefile']
        if file.filename == '':
            error_msg = "No file selected!"
            return render_template('result.html', error_msg=error_msg)

        img = Image.open(file.stream).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)

        # Predict
        out_pred, out_prob = predict(img)
        out_prob = round(out_prob * 100, 2)

        # Set danger level
        danger = "success" if "Normal" in out_pred else "danger"

        # Encode image to show on result page
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        encoded_img = base64.b64encode(img_io.read()).decode('ascii')

        return render_template('result.html', out_pred=out_pred, out_prob=out_prob, danger=danger, encoded_img=encoded_img)

    except Exception as e:
        error_msg = f"Image processing failed: {e}"
        return render_template('result.html', error_msg=error_msg)

# Prediction logic
def predict(img):
    if model is None:
        return "Model not loaded", 0.0

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    class_idx = np.argmax(pred, axis=1)[0]

    if class_idx == 0:
        result = ("Result: Brain Tumor\n"
                  "Symptoms: unexplained weight loss, double vision or vision loss, increased head pressure, dizziness, "
                  "balance issues, speech loss, hearing loss, or numbness on one side.")
    else:
        result = "You Are Safe, But Do keep precaution"

    return result, float(np.max(pred))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
