import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
from PIL import Image as PILImage, ImageDraw

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
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.open_button = QPushButton('Open Image', self)
        self.open_button.clicked.connect(self.open_image_file)
        self.open_button.setStyleSheet("font-size: 16px; padding: 10px;")
        
        # Layout setup
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        
        content_layout = QVBoxLayout()
        content_layout.addWidget(self.label)
        content_layout.addWidget(self.image_label)
        content_layout.addLayout(button_layout)
        
        container = QWidget()
        container.setLayout(content_layout)
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
        pil_image = pil_image.convert("RGB")
        
        # Draw a border around the detected disease
        draw = ImageDraw.Draw(pil_image)
        border_width = 10
        draw.rectangle([border_width, border_width, pil_image.width - border_width, pil_image.height - border_width], outline="red", width=border_width)
        
        pil_image = pil_image.resize((500, 500), PILImage.Resampling.LANCZOS)
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
