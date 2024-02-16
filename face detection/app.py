import os
from flask import Flask, request, render_template
import cv2

app = Flask(__name__)

# Function to detect faces in the image using image pyramid and sliding window
def detect_faces(image_path):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize a list to store detected faces
    detected_faces = []
    
    # Define the scale factor for the image pyramid
    scale_factor = 1.2
    
    # Loop over different scales of the image (image pyramid)
    for scale in [1, 1/scale_factor, scale_factor]:
        # Resize the image
        resized_image = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)))
        
        # Detect faces at the current scale
        faces = face_cascade.detectMultiScale(resized_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Scale the coordinates of detected faces back to the original image size
        for (x, y, w, h) in faces:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            
            # Filter out small or non-face regions based on size
            if w > 30 and h > 30:  # Adjust the threshold as needed
                detected_faces.append((x, y, w, h))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in detected_faces:
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
