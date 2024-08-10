import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image as PILImage

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

class SkinDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Skin Disease Detection')
        self.setGeometry(100, 100, 800, 600)
        
        # Create widgets
        self.label = QLabel('Detected Disease: None', self)
        self.image_label = QLabel(self)
        
        self.open_button = QPushButton('Open Image', self)
        self.open_button.clicked.connect(self.open_image_file)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.open_button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image_label)
        
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def open_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpg *.jpeg *.png *.bmp *.gif)')
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                detected_disease = detect_from_image(image)
                self.label.setText(f"Detected Disease: {detected_disease}")
                
                self.display_image(file_path)
            else:
                self.label.setText("Error: Unable to open image file.")
                
    def display_image(self, image_path):
        pil_image = PILImage.open(image_path)
        pil_image = pil_image.resize((250, 250), PILImage.Resampling.LANCZOS)
        qt_image = QImage(np.array(pil_image), pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    ex = SkinDiseaseApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
