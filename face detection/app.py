import os
from flask import Flask, request, render_template
import cv2
from mtcnn import MTCNN
import pandas as pd

app = Flask(__name__)

# Function to detect faces in the image using MTCNN
def detect_faces(image_path):
    # Load the MTCNN detector
    detector = MTCNN()
    
    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert the image to RGB format (if not already in RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    
    # Extract additional information about the detected faces
    face_details = []
    for result in faces:
        x, y, w, h = result['box']
        confidence = result['confidence']
        keypoints = result['keypoints']
        
        # Additional information such as confidence score and facial keypoints
        face_info = {
            'box': (x, y, w, h),
            'confidence': confidence,
            'keypoints': keypoints
        }
        
        face_details.append(face_info)
    
    # Draw rectangles around the detected faces
    for face_info in face_details:
        x, y, w, h = face_info['box']
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Save the output image with bounding boxes
    output_image_path = 'static/output.jpg'
    cv2.imwrite(output_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Save face details to an Excel file
    excel_file_path = 'static/face_details.xlsx'
    df = pd.DataFrame(face_details)
    df.to_excel(excel_file_path, index=False)
    
    return output_image_path, excel_file_path

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
            
            # Detect faces in the uploaded image and save details to Excel
            output_image_path, excel_file_path = detect_faces(file_path)
            
            # Display the results
            return render_template('index.html', image_file=output_image_path, excel_file=excel_file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
