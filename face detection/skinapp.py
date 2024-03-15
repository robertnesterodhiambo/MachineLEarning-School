import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('skin_disease_detection_model.h5')
target_size = (224, 224)  # Adjust based on your model's input shape

def preprocess_image(image):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=-1)
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image = Image.open(file)
            predicted_class = predict_image(image)
            return render_template('index.html', predicted_class=int(predicted_class))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
