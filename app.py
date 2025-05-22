# Streamlit App for Eye Disease Detection Using MobileNetV2

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# Define class names (update according to your dataset)
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Function to download model files from GitHub if they don't exist locally
def download_model_files():
    json_url = 'https://raw.githubusercontent.com/xZANDUBALMx/Eye-Disease-Detection/main/my_model.json'
    weights_url = 'https://raw.githubusercontent.com/xZANDUBALMx/Eye-Disease-Detection/main/my_model_weights.weights.h5'

    if not os.path.exists("my_model.json"):
        gdown.download(json_url, "my_model.json", quiet=False)

    if not os.path.exists("my_model_weights.weights.h5"):
        gdown.download(weights_url, "my_model_weights.weights.h5", quiet=False)

# Function to load model from downloaded files
@st.cache_resource
def load_model():
    download_model_files()
    with open("my_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("my_model_weights.weights.h5")
    return loaded_model

# Function to preprocess uploaded image
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Predict class and confidence
@st.cache_data
def predict_image(model, img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit UI
st.set_page_config(page_title="Eye Disease Detection", layout="centered")
st.title("👁️ Eye Disease Detection Using MobileNetV2")

st.markdown("""
Upload a retinal image and get the predicted eye condition:
- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal**
""")

# Load model once
model = load_model()

# Upload image section
uploaded_file = st.file_uploader("📤 Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Predict"):
        label, confidence = predict_image(model, image_data)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence}%")

# Optional: dataset access info for developers
with st.expander("ℹ️ Developer Info: Access Dataset"):
    st.markdown("""
    [📂 Download Dataset from Google Drive](https://drive.google.com/drive/folders/1qCsY10faS_8RQZ81bfitFJpgDxfwXBVE?usp=sharing)
    
    This dataset includes categorized images for training and testing:
    - `cataract`
    - `diabetic_retinopathy`
    - `glaucoma`
    - `normal`
    
    Note: Dataset is not required to use the app. Only for model training.
    """)
