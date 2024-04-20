from flask import Flask, request, jsonify
import cv2
import numpy as np
from mrcnn import utils
from mrcnn import model as modellib

app = Flask(__name__)

# Load pre-trained Mask R-CNN model
MODEL_DIR = "models"
MODEL_WEIGHTS_PATH = f"{MODEL_DIR}/pretrained_weights.h5"

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/detect_cracks', methods=['POST'])
def detect_cracks():
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run inference
    results = model.detect([image], verbose=1)

    # Format results
    # Example: For simplicity, just return the number of detected cracks
    num_cracks = len(results[0]['rois'])
    
    return jsonify({'result': f'Detected {num_cracks} crack(s)'})

if __name__ == '__main__':
    app.run(debug=True)
