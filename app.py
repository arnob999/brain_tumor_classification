from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Path to the model
MODEL_PATH = 'brain_tumor_2.h5'

# Load the model
model = load_model(MODEL_PATH)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a function to make predictions
def predict_tumor_type(img_array):
    predictions = model.predict(img_array)
    indices = predictions.argmax()

    tumor_types = {
        0: "Glioma",
        1: "Meningioma",
        2: "No Tumor",
        3: "Pituitary Tumor"
    }

    return tumor_types[indices]

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        
        # Ensure the file is present
        if not file:
            return "No file provided", 400

        # Save the file to ./uploads
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Preprocess the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (150, 150))
            img_array = np.array(img).reshape(1, 150, 150, 3)

            # Make prediction
            prediction = predict_tumor_type(img_array)

            # Pass relative file path to the template
            relative_file_path = os.path.join('uploads', filename).replace('\\', '/')

            return render_template('predict.html', prediction=prediction, image_path=relative_file_path)
        except Exception as e:
            return str(e), 500
    else:
        return render_template('upload.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
