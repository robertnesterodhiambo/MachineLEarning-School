import os
from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load pre-trained skin tone classification model from TensorFlow Hub
skin_tone_model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5")

# Function to classify skin tone in the image
def classify_skin_tone(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Resize image to match model input size (224x224)
    image = cv2.resize(image, (224, 224))
    
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image for model input
    image_preprocessed = image_rgb / 255.0  # Normalize pixel values
    
    # Expand dimensions to match model input shape
    image_input = np.expand_dims(image_preprocessed, axis=0)
    
    # Perform skin tone classification
    skin_tone_logits = skin_tone_model(image_input)
    
    # Convert logits to probabilities
    skin_tone_probabilities = tf.nn.softmax(skin_tone_logits)
    
    # Get the predicted skin tone category
    predicted_skin_tone = np.argmax(skin_tone_probabilities)
    
    # Define skin tone categories (example)
    skin_tone_categories = ["Light", "Medium", "Dark"]
    
    # Get the predicted skin tone category label
    predicted_skin_tone_label = skin_tone_categories[predicted_skin_tone]
    
    return predicted_skin_tone_label

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
            # Save the uploaded file
            file_path = 'static/uploaded_image.jpg'
            file.save(file_path)
            
            # Classify skin tone in the uploaded image
            predicted_skin_tone = classify_skin_tone(file_path)
            
            # Display the results
            return render_template('index.html', message=f'Predicted skin tone: {predicted_skin_tone}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
