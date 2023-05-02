"""
                                    Flask app
                                    By Prasant Poudel

"""


from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import cv2

app = Flask(__name__)

# Define list of class names for your custom dataset
class_names = {
    0: "Actinic Keratoses and Intraepithelial Carcinomae (Cancer)",
    1: "Basal Cell Carcinoma (Cancer)",
    2: "Benign Keratosis-like Lesions (Non-Cancerous)",
    3: "Dermatofibroma (Non-Cancerous)",
    4: "Melanoma (Cancer)",
    5: "Melanocytic Nevi (Non-Cancerous)",
    6: "Vascular Lesion (Non-Cancerous)"
}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model/InceptionResNetV2Skripsi.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to preprocess the image before prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x

def detect_skin(image):
    # Convert the image to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Apply skin color detection algorithm
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Apply morphological transformations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Count the number of skin pixels
    num_skin_pixels = cv2.countNonZero(mask)
    
    # Calculate the ratio of skin pixels to total pixels
    ratio = num_skin_pixels / (image.shape[0] * image.shape[1])
    
    return ratio

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')    
def about():
    return render_template('info.html')

@app.route('/prevention')    
def prevention():
    return render_template('prevention.html')

@app.route('/riskFactor')    
def riskFactor():
    return render_template('riskFactor.html')

@app.route('/earlyDetection')    
def earlyDetection():
    return render_template('earlyDetection.html')

@app.route('/classify')    
def classify():
    return render_template('classify.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load image to predict
    img_file = request.files['image']
    img_path = 'static/uploads/' + img_file.filename
    img_file.save(img_path)
    
     # Load image and perform skin detection
    img = cv2.imread(img_path)
    skin_ratio = detect_skin(img)

    # If skin ratio is below threshold, return "Not a valid skin image"
    if skin_ratio < 0.1:
        result = {'class_name': "Not a valid skin image"}
        return render_template('result.html', result=result, img_path=img_path)
    
    # Preprocess image and make prediction
    x = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    class_idx = np.argmax(preds[0])
    class_name = class_names[class_idx]
    
    # Return prediction result
    result = {'class_name': class_name}
    return render_template('result.html', result=result, img_path=img_path)

if __name__ == '__main__':
    app.run(port=8080,debug=True) 

