import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os
import gdown

# Define class names (update as per your training labels)
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Function to download model files
def download_model_files():
    json_url = 'https://raw.githubusercontent.com/xZANDUBALMx/Eye-Disease-Detection/main/my_model.json'
    weights_url = 'https://raw.githubusercontent.com/xZANDUBALMx/Eye-Disease-Detection/main/my_model_weights.weights.h5'

    if not os.path.exists("my_model.json"):
        gdown.download(json_url, "my_model.json", quiet=False)

    if not os.path.exists("my_model_weights.weights.h5"):
        gdown.download(weights_url, "my_model_weights.weights.h5", quiet=False)

@st.cache_resource
def load_model():
    download_model_files()
    with open("my_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("my_model_weights.weights.h5")
    return loaded_model

# Image preprocessing
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Prediction function
@st.cache_data
def predict_image(model, img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    predicted_class = class_names[np.argmax(probabilities)]
    confidence = round(100 * np.max(probabilities), 2)
    return predicted_class, confidence

# Streamlit UI
st.set_page_config(page_title="Eye Disease Detection", layout="centered")
st.title("\U0001F441\uFE0F Eye Disease Detection Using MobileNetV2")

st.markdown("""
Upload a retinal image and get the predicted eye condition:
- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal**
""")

model = load_model()

uploaded_file = st.file_uploader("\U0001F4C4 Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_container_width=True)

    if st.button("\U0001F9E0 Predict"):
        label, confidence = predict_image(model, image_data)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence}%")

with st.expander("\u2139\uFE0F Developer Info: Access Dataset"):
    st.markdown("""
    [\U0001F4C2 Download Dataset from Google Drive](https://drive.google.com/drive/folders/1qCsY10faS_8RQZ81bfitFJpgDxfwXBVE?usp=sharing)

    This dataset includes categorized images for training and testing:
    - `cataract`
    - `diabetic_retinopathy`
    - `glaucoma`
    - `normal`
    """)
