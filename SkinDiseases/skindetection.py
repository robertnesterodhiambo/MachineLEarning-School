import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model_path = '/home/dragon/Git/model/best_skin_disease_model.h5'
model = load_model(model_path)

# Load class labels from the training data
data_dir = os.path.expanduser('~/Git/MachineLEarning-School/SkinDiseases/skin-disease-datasaet')
train_dir = os.path.join(data_dir, 'train_set')
class_labels = sorted(os.listdir(train_dir))

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to detect skin disease from an image
def detect_from_image(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

# Function to detect skin disease from live video feed
def detect_from_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_image = preprocess_image(frame)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]

        # Display prediction on the frame
        cv2.putText(frame, f'Detected: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Skin Disease Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Command-line interface to choose mode
def main():
    print("Choose mode:")
    print("1. Upload an Image")
    print("2. Live Camera Detection")

    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        image_path = input("Enter the path to the image: ")
        detected_disease = detect_from_image(image_path)
        print(f"Detected Disease: {detected_disease}")

    elif choice == '2':
        detect_from_video()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
