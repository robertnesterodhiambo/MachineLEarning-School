import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

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
def detect_from_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

# Function to open image file
def open_image_file():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if file_path:
            if not os.path.isfile(file_path):
                messagebox.showerror("Error", "The selected file does not exist.")
                return

            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Unable to open the image file.")
                return

            detected_disease = detect_from_image(image)
            result_label.config(text=f"Detected Disease: {detected_disease}")

            # Display the image in Tkinter window
            display_image(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to display the image in Tkinter window
def display_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((250, 250), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)

        # Create or update the image label
        if hasattr(display_image, 'img_label'):
            display_image.img_label.config(image=img)
            display_image.img_label.image = img
        else:
            display_image.img_label = tk.Label(root, image=img)
            display_image.img_label.image = img
            display_image.img_label.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while displaying the image: {e}")

# Function to start live video detection
def start_video_detection():
    global video_running
    video_running = True
    video_thread = threading.Thread(target=video_detection)
    video_thread.start()

# Function to stop live video detection
def stop_video_detection():
    global video_running
    video_running = False

# Function to detect skin disease from live video feed
def video_detection():
    global cap
    cap = cv2.VideoCapture(0)
    while video_running:
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

# Create the Tkinter app
root = tk.Tk()
root.title("Skin Disease Detection")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

open_image_button = tk.Button(frame, text="Upload Image", command=open_image_file)
open_image_button.pack(side=tk.LEFT, padx=5)

start_video_button = tk.Button(frame, text="Start Video Detection", command=start_video_detection)
start_video_button.pack(side=tk.LEFT, padx=5)

stop_video_button = tk.Button(frame, text="Stop Video Detection", command=stop_video_detection)
stop_video_button.pack(side=tk.LEFT, padx=5)

result_label = tk.Label(root, text="Detected Disease: None")
result_label.pack(pady=10)

video_running = False

root.mainloop()
