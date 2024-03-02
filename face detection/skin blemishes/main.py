from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'  # Create a separate folder for result images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
model_path = 'best.pt'

# Load the pre-trained YOLOv5 model
model = YOLO(model_path)

def inference(image_path, result_folder):
    # Run inference on a single image
    results = model(image_path)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.save(filename=os.path.join(result_folder, 'result.jpg'))  # Save with a specific name

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Perform inference on the uploaded image and save with a specific name
        inference(file_path, RESULT_FOLDER)

        # Optionally, you can return a response or redirect to a different page
        return redirect(url_for('result'))

    return redirect(url_for('index'))

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/display_result')
def display_result():
    result_image_path = os.path.join(RESULT_FOLDER, 'result.jpg')
    return send_file(result_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
