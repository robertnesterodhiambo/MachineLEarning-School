from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define model-related parameters
input_shape = (224, 224, 3)  # Input shape of the model
num_classes = 2  # Number of output classes

# Define a function to modify the model architecture
def modify_model_architecture(original_model):
    # Create a new model with modified architecture
    input_layer = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Replace the incompatible DepthwiseConv2D layer with a compatible Conv2D layer
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Construct the modified model
    modified_model = Model(inputs=input_layer, outputs=output_layer)

    return modified_model

# Load the original model architecture from the saved model file
original_model = load_model('/home/oem/repos/MachineLEarning-School/face detection/skin_disease_detection_model.h5', compile=False)

# Modify the model architecture
modified_model = modify_model_architecture(original_model)

# Load the model weights into the modified architecture
modified_model.load_weights('/home/oem/repos/MachineLEarning-School/face detection/skin_disease_detection_model_weights.h5')

# Compile the modified model if needed
# modified_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define function to process the uploaded image
def process_image(image_path):
    img = Image.open(image_path)
    # Preprocess the image (resize, normalize, etc.)
    # Example:
    img = img.resize((input_shape[0], input_shape[1]))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define route for skin disease detection
@app.route('/detect_skin_disease', methods=['POST'])
def detect_skin_disease():
    # Check if request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})
    
    # Check if file is supported image format
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Unsupported file format'})
    
    try:
        # Save the uploaded image temporarily
        # Example:
        file_path = 'temp.jpg'
        file.save(file_path)
        
        # Process the uploaded image
        processed_img = process_image(file_path)
        
        # Perform skin disease detection using the modified model
        prediction = modified_model.predict(processed_img)
        
        # Decode the prediction into human-readable format (e.g., disease name)
        # Example:
        disease_name = "Melanoma" if prediction[0][0] > prediction[0][1] else "Non-melanoma"
        
        # Return the result in JSON format
        return jsonify({'disease_name': disease_name})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
