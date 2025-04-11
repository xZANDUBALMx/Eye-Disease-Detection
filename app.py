import streamlit as st
import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ====================================================
# 1) Global Constants and Settings
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Update these dimensions to those your model was trained with.
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Diagnostic classes mapping (keys must match your model's encoding)
DIAGNOSIS_DICT = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# ====================================================
# 2) Load the Model Once (Cached)
# ====================================================
@st.cache_resource
def load_dr_model():
    model_path = os.path.join(BASE_DIR, 'efficientnetb1.keras')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

# ====================================================
# 3) Image Preprocessing (Manual Method)
# ====================================================
def preprocess_image(image_input):
    try:
        # Open image and ensure RGB.
        image = Image.open(image_input).convert('RGB')
        # Resize to the dimensions expected by the model.
        image = ImageOps.fit(image, (IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)
        # Use the same normalization as in training.
        normalized_image_array = (image_array / 127.0) - 1
        data = np.expand_dims(normalized_image_array, axis=0)
        st.write("DEBUG: Preprocessed image shape:", data.shape)
        st.write("DEBUG: Preprocessed image min/max:", data.min(), data.max())
        return data
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# ====================================================
# 4) Model Prediction Function
# ====================================================
def dr_prediction(image_input):
    data = preprocess_image(image_input)
    if data is None:
        return None, None, None

    model = load_dr_model()
    st.write("DEBUG: Model expected input shape:", model.input_shape)
    st.write("DEBUG: Data shape for prediction:", data.shape)
    try:
        prediction = model.predict(data)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None, None, None

    st.write("DEBUG: Raw prediction vector:", prediction[0])
    pred_index = np.argmax(prediction, axis=-1)[0]
    confidence = prediction[0][pred_index] * 100
    return DIAGNOSIS_DICT[pred_index], confidence, prediction

# ====================================================
# 5) Find the Image's Folder in the Dataset
# ====================================================
def find_image_dataset_folder(filename):
    """
    Searches the gaussian_filtered_images folder (and its subfolders)
    for the given filename. Returns the folder name if found.
    """
    dataset_root = os.path.join(BASE_DIR, "gaussian_filtered_images")
    # Loop through each subfolder of gaussian_filtered_images.
    for subfolder in os.listdir(dataset_root):
        subfolder_path = os.path.join(dataset_root, subfolder)
        if os.path.isdir(subfolder_path):
            if filename in os.listdir(subfolder_path):
                return subfolder  # This is the class folder
    return None

# ====================================================
# 6) Plot Prediction Distribution (for a single prediction)
# ====================================================
def plot_prediction_distribution(prediction):
    classes = list(DIAGNOSIS_DICT.values())
    confidences = prediction[0]
    fig, ax = plt.subplots()
    ax.bar(classes, confidences, color='skyblue')
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence Distribution')
    return fig

# ====================================================
# 7) Analyze the Dataset Folder (of the predicted image)
# ====================================================
def analyze_dataset(pred_class):
    dataset_folder = os.path.join(BASE_DIR, "gaussian_filtered_images", pred_class)
    if not os.path.exists(dataset_folder):
        st.error(f"Dataset folder for class '{pred_class}' not found at: {dataset_folder}")
        return

    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        st.write("No images found in the predicted class folder.")
        return

    st.subheader("Sample Images from Dataset Folder")
    sample_count = min(25, len(image_files))
    sampled_files = np.random.choice(image_files, sample_count, replace=False)
    cols = st.columns(5)
    for i, img_path in enumerate(sampled_files):
        with cols[i % 5]:
            st.image(img_path, use_column_width=True)

    st.subheader("Prediction Distribution in Dataset Folder")
    model = load_dr_model()
    predictions_list = []
    num_images_to_predict = min(100, len(image_files))
    for img_path in image_files[:num_images_to_predict]:
        try:
            with Image.open(img_path).convert('RGB') as img:
                img = ImageOps.fit(img, (IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
                img_array = np.asarray(img).astype(np.float32)
                normalized_img = (img_array / 127.0) - 1
                data = np.expand_dims(normalized_img, axis=0)
                pred = model.predict(data)
                pred_index = np.argmax(pred, axis=-1)[0]
                predictions_list.append(pred_index)
        except Exception as e:
            st.write(f"Error processing {img_path}: {e}")

    if predictions_list:
        counts = {}
        for pred in predictions_list:
            counts[pred] = counts.get(pred, 0) + 1
        sorted_keys = sorted(counts.keys())
        pred_classes = [DIAGNOSIS_DICT.get(i, str(i)) for i in sorted_keys]
        pred_counts = [counts[i] for i in sorted_keys]
        fig, ax = plt.subplots()
        ax.bar(pred_classes, pred_counts, color='skyblue')
        ax.set_title("Prediction Distribution in Folder")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.write("No predictions generated for analysis.")

# ====================================================
# 8) Main Streamlit App
# ====================================================
def main():
    st.title("DetAll: Diabetic Retinopathy Analysis from Dataset Images")
    st.sidebar.title("Navigation")
    menu_options = ["Home", "Dataset Analysis"]
    choice = st.sidebar.selectbox("Menu", menu_options)

    if choice == "Home":
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text("Initializing the DR analysis tool...")
            time.sleep(0.01)
        status_text.success("Ready!")
        st.write("This application analyzes diabetic retinopathy images from the dataset. "
                 "Please upload an image from your dataset folder (gaussian_filtered_images).")

    elif choice == "Dataset Analysis":
        st.header("Dataset Image Analysis")
        st.write("Upload an image (from your gaussian_filtered_images folder) to see which class it belongs to and review analysis based on that folder.")
        image_input = st.file_uploader("Upload dataset image", type=["jpg", "png", "jpeg"])
        if image_input is not None:
            # Show the uploaded image.
            uploaded_image = Image.open(image_input).convert('RGB')
            st.image(uploaded_image, caption="Uploaded Image for Analysis", use_column_width=True)

            # Identify the dataset folder based on the filename.
            filename = image_input.name
            dataset_folder = find_image_dataset_folder(filename)
            if dataset_folder is None:
                st.error("Could not locate the uploaded image in the dataset (gaussian_filtered_images).")
            else:
                st.info(f"Image found in dataset folder: **{dataset_folder}**")

            # Run model prediction on the uploaded image.
            pred_class, confidence, prediction = dr_prediction(image_input)
            if pred_class is not None:
                st.success(f"Model Prediction: **{pred_class}** with {confidence:.2f}% confidence.")
                st.write("Raw prediction vector:", prediction[0])
                fig = plot_prediction_distribution(prediction)
                st.pyplot(fig)
            else:
                st.error("Model prediction failed. Check debug details above.")

            # Now, run analysis on the dataset folder corresponding to this image.
            # If the model prediction (or the found folder) indicates a class, analyze that folder.
            class_to_analyze = dataset_folder if dataset_folder is not None else pred_class
            if class_to_analyze:
                analyze_dataset(class_to_analyze)
    
    # ====================================================
    # Hide default Streamlit UI elements
    # ====================================================
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
