import os
from flask import Flask, request, render_template
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import base64
import urllib

app = Flask(__name__)

# Load your trained model
model = load_model('save.h5')  # Ensure the model path is correct

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')    

@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    print("Image Upload Requested")  # Debugging print
    try:
        img = Image.open(request.files['imagefile'].stream).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    except Exception as e:
        print(f"Error in image processing: {e}")  # Debugging print
        error_msg = "Please choose an image file!"
        return render_template('result.html', **locals())

    # Call Function to predict
    args = {'input': img}
    out_pred, out_prob = predict(args)
    out_prob = out_prob * 100  # Convert to percentage
    print(f"Prediction: {out_pred}, Probability: {out_prob}")  # Debugging print

    danger = "danger"
    if out_pred == "You Are Safe, But Do keep precaution":
        danger = "success"
    
    # Prepare image for display
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    png_output = base64.b64encode(img_io.getvalue())
    processed_file = urllib.parse.quote(png_output)

    return render_template('result.html', out_pred=out_pred, out_prob=out_prob, processed_file=processed_file, danger=danger)

def predict(args):
    # Convert image to numpy array and normalize
    img = np.array(args['input']) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the loaded model
    pred = model.predict(img)

    # Check the prediction result
    if np.argmax(pred, axis=1)[0] == 0:
        out_pred = "Result: Brain Tumor  Symptoms: unexplained weight loss, double vision, hearing loss, weakness on one side of the body."
    elif np.argmax(pred, axis=1)[0] == 1:
        out_pred = "Result: Normal"
    else:
        out_pred = "Unknown result"
    
    return out_pred, float(np.max(pred))  # Return prediction and max probability

if __name__ == '__main__':
    app.run(debug=True)
