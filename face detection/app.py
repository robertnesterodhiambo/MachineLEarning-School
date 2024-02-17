import os
from flask import Flask, request, render_template
import cv2
from mtcnn import MTCNN

app = Flask(__name__)

# Function to detect faces in the image using MTCNN
def detect_faces(image_path):
    # Load the MTCNN detector
    detector = MTCNN()
    
    # Read the input image
    image = cv2.imread(image_path)
    
    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    # Draw rectangles around the detected faces
    for result in faces:
        x, y, w, h = result['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Save the output image with bounding boxes
    output_image_path = 'static/output.jpg'
    cv2.imwrite(output_image_path, image)
    
    return output_image_path

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
            
            # Detect faces in the uploaded image
            output_image_path = detect_faces(file_path)
            
            # Display the results
            return render_template('index.html', image_file=output_image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
