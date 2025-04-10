import streamlit as st
import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image, ImageOps

# Define the base directory as the directory where app.py is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DR Diagnosis Mapping
DIAGNOSIS_DICT = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# Cache the loading of the model so it doesn't reload every time
@st.cache_resource
def load_dr_model():
    model_path = os.path.join(BASE_DIR, 'efficientnetb1.keras')
    model = tf.keras.models.load_model(model_path)
    return model

def dr_prediction(image_input):
    # Prepare the image for prediction.
    size = (224, 224)
    image = Image.open(image_input)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    
    model = load_dr_model()
    prediction = model.predict(data)
    pred_class = np.argmax(prediction, axis=-1)[0]
    confidence = prediction[0][pred_class] * 100
    return DIAGNOSIS_DICT[pred_class], confidence

def main():
    st.title("DetAll: Diabetic Retinopathy Detection")
    st.sidebar.title("Navigation")
    menu_options = ["Home", "DR Detection"]
    choice = st.sidebar.selectbox("Menu", menu_options)

    if choice == "Home":
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text("Setting up the DR detection tool...")
            time.sleep(0.01)
        status_text.success("Ready!")
        st.write(
            "This application detects diabetic retinopathy from eye images "
            "using a pre-trained EfficientNetB1 model."
        )

    elif choice == "DR Detection":
        st.sidebar.write("Upload an eye image for diabetic retinopathy detection.")
        image_input = st.sidebar.file_uploader("Choose an eye image", type=["jpg", "png"])
        if image_input is not None:
            # Display the uploaded image.
            image_bytes = image_input.read()
            display_size = st.slider("Adjust displayed image size:", 300, 1000, 500)
            st.image(image_bytes, width=display_size, caption="Uploaded Image")
            # Rewind the file pointer so it can be read again
            image_input.seek(0)
            if st.sidebar.button("Analyze DR"):
                with st.spinner("Processing..."):
                    pred_class, confidence = dr_prediction(image_input)
                    st.success(
                        f"Prediction: **{pred_class}** with {confidence:.2f}% confidence."
                    )

if __name__ == '__main__':
    main()

# Optional: Hide default Streamlit style elements.
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
