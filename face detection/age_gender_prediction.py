import cv2
from mtcnn import MTCNN
from keras.models import load_model
import numpy as np

# Load the MTCNN detector
detector = MTCNN()

# Load pre-trained age and gender prediction models
age_model = load_model('age_model.h5')  # Replace 'age_model.h5' with the path to your age prediction model
gender_model = load_model('gender_model.h5')  # Replace 'gender_model.h5' with the path to your gender prediction model

# Function to preprocess face image for age and gender prediction
def preprocess_image(face_image):
    # Resize the face image to match the input size expected by the models
    resized_image = cv2.resize(face_image, (224, 224))
    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image.astype('float32') / 255.0
    # Expand dimensions to create a batch of size 1
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to predict age and gender
def predict_age_gender(face_image):
    # Detect faces in the image
    faces = detector.detect_faces(face_image)
    
    if len(faces) == 0:
        return None, None  # No face detected
    
    # Get the bounding box coordinates of the first face detected
    x, y, w, h = faces[0]['box']
    # Extract the face region from the image
    face_roi = face_image[y:y+h, x:x+w]
    
    # Preprocess the face image for age and gender prediction
    input_image = preprocess_image(face_roi)

    # Predict age
    age_prediction = age_model.predict(input_image)[0][0] * 100  # Convert to years
    
    # Predict gender
    gender_prediction = gender_model.predict(input_image)[0][0]
    gender = 'Male' if gender_prediction < 0.5 else 'Female'
    
    return age_prediction, gender
