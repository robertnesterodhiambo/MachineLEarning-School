from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = '/home/dragon/repos/MachineLEarning-School/skin blemishes/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = '/home/dragon/repos/MachineLEarning-School/skin blemishes/best.pt'

# Load the pre-trained YOLOv5 model
model = YOLO(MODEL_PATH)

def inference(image_path, result_folder):
    try:
        # Run inference on a single image
        results = model(image_path)
        print(f"Number of results: {len(results)}")

        with open(os.path.join(result_folder, 'disease_names.txt'), 'w') as txt_file:
            for i, result in enumerate(results):
                print(f"Processing result {i}")
                result_image_path = os.path.join(result_folder, f'result_{i}.jpg')
                result.save(filename=result_image_path)

                detected_diseases = []

                # Check if the Results object has 'names' attribute
                if hasattr(result, 'names'):
                    for det in result.names:
                        # Extract class label and confidence
                        class_label = det['name']
                        confidence = det['confidence']

                        # Assume class label contains disease name
                        disease_name = class_label
                        detected_diseases.append(disease_name)
                else:
                    print("No 'names' attribute found in the Results object.")
                    txt_file.write(f"No detection results found.\n")
                    continue

                txt_file.write(f"Image Path: {result_image_path}\n")
                txt_file.write(f"Detected Diseases: {detected_diseases}\n")

                # Open the result image and extract metadata
                with Image.open(result_image_path) as img:
                    txt_file.write(f"Image Size: {img.size}\n")
                    txt_file.write(f"Image Format: {img.format}\n")

    except Exception as e:
        print(f"Error during inference: {e}")

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
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Perform inference on the uploaded image
            inference(file_path, RESULT_FOLDER)

            # Redirect to the result page
            return redirect(url_for('result'))
        except Exception as e:
            print(f"Error during file upload or inference: {e}")
            return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/display_result')
def display_result():
    result_image_path = os.path.join(RESULT_FOLDER, 'result_0.jpg')
    return send_file(result_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
