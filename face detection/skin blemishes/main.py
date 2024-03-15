from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = '/home/oem/repos/MachineLEarning-School/face detection/skin blemishes/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = '/home/oem/repos/MachineLEarning-School/face detection/skin blemishes/best.pt'

# Load the pre-trained YOLOv5 model
model = YOLO(MODEL_PATH)

def inference(image_path, result_folder):
    try:
        # Run inference on a single image
        results = model(image_path)
        result_data = []

        for i, result in enumerate(results):
            result_image_path = os.path.join(result_folder, f'result_{i}.jpg')
            result.save(filename=result_image_path)
            result_info = {
                "image_path": result_image_path,
                "detections": []
            }

            for det in result.pred:
                # Extract class label and confidence
                class_label = model.names[int(det[-1])]
                confidence = det[-2]

                # Store detection information
                detection_info = {
                    "class": class_label,
                    "confidence": confidence,
                    "box": det[:4].tolist()  # Convert to list to serialize in JSON
                }
                result_info["detections"].append(detection_info)

            result_data.append(result_info)

        # Serialize the result data to JSON
        json_data = json.dumps(result_data, indent=4)
        with open(os.path.join(result_folder, 'result.json'), 'w') as json_file:
            json_file.write(json_data)

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
