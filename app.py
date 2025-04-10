import streamlit as st
import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import datetime

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
    pred_index = np.argmax(prediction, axis=-1)[0]
    confidence = prediction[0][pred_index] * 100
    return DIAGNOSIS_DICT[pred_index], confidence, prediction

def plot_prediction_distribution(prediction):
    # Plot a bar chart for the prediction distribution.
    classes = list(DIAGNOSIS_DICT.values())
    confidences = prediction[0]
    fig, ax = plt.subplots()
    ax.bar(classes, confidences, color='skyblue')
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence Distribution')
    return fig

def save_uploaded_file(uploaded_file):
    # Create an "uploads" directory if it doesn't exist
    uploads_dir = os.path.join(BASE_DIR, "uploads")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    # Create a unique folder with a timestamp
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = os.path.join(uploads_dir, time_stamp)
    os.makedirs(new_folder, exist_ok=True)
    # Define saved file name (keeping its original extension)
    file_ext = os.path.splitext(uploaded_file.name)[1]
    save_path = os.path.join(new_folder, f"uploaded{file_ext}")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def main():
    st.title("DetAll: Diabetic Retinopathy Detection")
    st.sidebar.title("Navigation")
    menu_options = ["Home", "DR Detection", "Details"]
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
            # Save the uploaded image in a new unique folder.
            saved_path = save_uploaded_file(image_input)
            st.write(f"Image successfully saved at: **{saved_path}**")
            # Display the saved image.
            saved_image = Image.open(saved_path)
            display_size = st.slider("Adjust displayed image size:", 300, 1000, 500)
            st.image(saved_image, width=display_size, caption="Uploaded Image")
            # Rewind the file pointer so it can be read again for prediction.
            image_input.seek(0)
            if st.sidebar.button("Analyze DR"):
                with st.spinner("Processing..."):
                    pred_class, confidence, prediction = dr_prediction(image_input)
                    st.success(
                        f"Prediction: **{pred_class}** with {confidence:.2f}% confidence."
                    )
                    # Plot and display the prediction distribution.
                    fig = plot_prediction_distribution(prediction)
                    st.pyplot(fig)
    
    elif choice == "Details":
        st.header("Detailed Visualizations")
        st.write(
            "Below are the supplementary graphs and visualizations that provide "
            "more insights into the model's performance and training process."
        )
        # Define the folder where your detailed images are stored.
        details_folder = os.path.join(BASE_DIR, "details")
        if os.path.exists(details_folder):
            # Get a sorted list of image files in the folder.
            detail_images = sorted([f for f in os.listdir(details_folder)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if detail_images:
                for img_file in detail_images:
                    img_path = os.path.join(details_folder, img_file)
                    st.image(img_path, caption=img_file, use_column_width=True)
            else:
                st.write("No image files found in the 'details' folder.")
        else:
            st.write(
                "The 'details' folder does not exist in the project directory. "
                "Please create a folder named 'details' and add your visualization images there."
            )

    # Optional: Hide default Streamlit style elements.
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
