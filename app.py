import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.models import model_from_json

MODEL_CONFIG_PATH = 'app/model/config.json'
MODEL_WEIGHTS_PATH = 'app/model/model.weights.h5'

classes = {
    0: 'Sugar beet',
    1: 'Common wheat',
    2: 'Charlock',
    3: 'Cleavers',
    4: 'Common Chickweed',
    5: 'Shepherds Purse',
    6: 'Maize',
    7: 'Scentless Mayweed',
    8: 'Loose Silky-bent',
    9: 'Fat Hen',
    10: 'Black-grass',
    11: 'Small-flowered Cranesbill'
}

# Load model from config.json and model.weights.h5
def load_model_from_files(config_path, weights_path):
    with open(config_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# Initialize model
model = load_model_from_files(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
print("Model loaded successfully!")

#===========================================================================

def create_mask_for_plant(image):

    print(image.shape)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

# Combine all processing steps
def segment(img):
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    return image_sharpen

#===========================================================================

# Set title of the app
st.title("Plant Seedlings Classification")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Check if a file is uploaded
if uploaded_file is not None:


    st.image(uploaded_file, caption="Uploaded Image", use_container_width =True)
    
    # Load and process the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert the image from RGB (PIL) to BGR (OpenCV format)
    print(img_array.shape)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(150,150))
    # Segment and sharpen image before prediction
    img_processed = segment(img)
    print(img_processed.shape)

    # Ensure the image has the correct shape for the model input
    img_processed = np.expand_dims(img_processed, axis=0)  # Add batch dimension
    img_processed = img_processed.astype('float32') / 255.0 
    
    # Make prediction using the model
    predictions = model.predict([img_processed])

    # Get the predicted class and confidence level
    class_index = np.argmax(predictions)
    class_name = classes.get(class_index, "Unknown")
    confidence = float(predictions[0][class_index])

    # Centered result panel with Markdown/HTML
    result_html = f"""
    <div style="text-align: center;">
        <h3>Prediction: {class_name}</h3>
        <p>Confidence: {confidence * 100:.2f}%</p>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)

