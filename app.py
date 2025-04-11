import streamlit as st
import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import pandas as pd
import cv2
import shutil

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# 1. GLOBAL CONSTANTS AND PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust dimensions based on your training pipeline.
IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Model file (assumed saved in the repo)
MODEL_PATH = os.path.join(BASE_DIR, 'efficientnetb1.keras')

# Folder in which your dataset images are stored.
DATASET_DIR = os.path.join(BASE_DIR, "gaussian_filtered_images")

# CSV file with training information (assumed in repo root)
CSV_PATH = os.path.join(BASE_DIR, "train.csv")

# Mapping (ensure keys match your model encoding)
DIAGNOSIS_DICT = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# =============================================================================
# 2. MODEL LOADING & PREDICTION FUNCTIONS
# =============================================================================
@st.cache_resource
def load_dr_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image_input):
    try:
        # Ensure the image is RGB and resize it.
        image = Image.open(image_input).convert('RGB')
        image = ImageOps.fit(image, (IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)
        # Normalize in the same way as during training.
        normalized_image_array = (image_array / 127.0) - 1
        data = np.expand_dims(normalized_image_array, axis=0)
        st.write("DEBUG: Preprocessed image shape:", data.shape)
        st.write("DEBUG: Preprocessed image min/max:", data.min(), data.max())
        return data
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

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

def plot_prediction_distribution(prediction):
    # Plot a bar chart for the prediction vector.
    classes = list(DIAGNOSIS_DICT.values())
    confidences = prediction[0]
    fig, ax = plt.subplots()
    ax.bar(classes, confidences, color='skyblue')
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence Distribution')
    return fig

# =============================================================================
# 3. DATASET IMAGE LOOKUP & ANALYSIS FUNCTIONS
# =============================================================================
def find_image_dataset_folder(filename):
    """
    Search through the dataset folder (with subdirectories by class)
    for the given filename.
    """
    if not os.path.exists(DATASET_DIR):
        st.error(f"Dataset directory not found: {DATASET_DIR}")
        return None
    for subfolder in os.listdir(DATASET_DIR):
        subfolder_path = os.path.join(DATASET_DIR, subfolder)
        if os.path.isdir(subfolder_path):
            if filename in os.listdir(subfolder_path):
                return subfolder  # Return the class folder name
    return None

def analyze_dataset(pred_class):
    """
    Analyze the dataset folder corresponding to a given class.
    Displays a sample of images and a prediction distribution (by running model.predict
    on a subset of images in that folder).
    """
    folder_path = os.path.join(DATASET_DIR, pred_class)
    if not os.path.exists(folder_path):
        st.error(f"Dataset folder for class '{pred_class}' not found at: {folder_path}")
        return
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
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

# =============================================================================
# 4. MODEL EVALUATION FUNCTIONS (Using CSV and Test Generator)
# =============================================================================
@st.cache_data
def load_train_dataframe():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found at: {CSV_PATH}")
        return None
    df = pd.read_csv(CSV_PATH)
    # Map your diagnosis numbers to labels
    diagnosis_dict_binary = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    diagnosis_dict = DIAGNOSIS_DICT  # same mapping for class names
    df['binary_type'] = df['diagnosis'].map(diagnosis_dict_binary.get)
    df['labels'] = df['diagnosis'].map(diagnosis_dict.get)
    # Build filepaths: assume images are in gaussian_filtered_images/<label>/filename.png
    filepaths = []
    labels = []
    for label in df['labels'].unique():
        folder = os.path.join(DATASET_DIR, label)
        if os.path.isdir(folder):
            flist = os.listdir(folder)
            for f in flist:
                filepaths.append(os.path.join(folder, f))
                labels.append(label)
    df_data = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    return df_data

def create_test_generator(df, batch_size=40):
    # Use a simple preprocessing that rescales images
    tvgen = ImageDataGenerator(preprocessing_function=lambda img: img)
    test_gen = tvgen.flow_from_dataframe(
        df, x_col='filepaths', y_col='labels',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        color_mode='rgb', shuffle=False, batch_size=batch_size)
    return test_gen

def display_model_evaluation():
    st.subheader("Model Evaluation on Test Set")
    df_data = load_train_dataframe()
    if df_data is None:
        return
    # Split data into train, valid, test (here we use 80% train, 10% valid, 10% test)
    train_df, dummy_df = train_test_split(df_data, train_size=0.8, shuffle=True, random_state=123)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)
    st.write(f"Test set length: {len(test_df)}")
    
    test_gen = create_test_generator(test_df, batch_size=32)
    model = load_dr_model()
    with st.spinner("Predicting on test set..."):
        preds = model.predict(test_gen)
    # Build y_true and y_pred from test_gen
    y_true = np.array(test_gen.labels)
    y_pred = np.array([np.argmax(pred) for pred in preds])
    
    # Plot confusion matrix
    class_indices = test_gen.class_indices
    classes = [key for key, val in sorted(class_indices.items(), key=lambda x: x[1])]
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
    fig_cm, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig_cm)
    
    # Display classification report
    clr = classification_report(np.argmax(y_true, axis=1), y_pred, target_names=classes, output_dict=True)
    clr_text = classification_report(np.argmax(y_true, axis=1), y_pred, target_names=classes)
    st.text("Classification Report:")
    st.text(clr_text)

# =============================================================================
# 5. STREAMLIT APP MAIN
# =============================================================================
def main():
    st.title("DetAll: Diabetic Retinopathy Analysis")
    st.sidebar.title("Navigation")
    menu_options = ["Home", "Dataset Analysis", "Model Evaluation"]
    choice = st.sidebar.selectbox("Select Tab", menu_options)
    
    if choice == "Home":
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text("Initializing DR analysis tool...")
            time.sleep(0.005)
        status_text.success("Ready!")
        st.write("Welcome! This app analyzes diabetic retinopathy images from your dataset.\n"
                 "Use the **Dataset Analysis** tab to upload an image and see its predicted class and "
                 "analysis based on the dataset folder. Use **Model Evaluation** to view evaluation metrics "
                 "from a held-out test set.")

    elif choice == "Dataset Analysis":
        st.header("Dataset Image Analysis")
        st.write("Upload an image (from your gaussian_filtered_images folder) to determine its class and view analysis.")
        image_input = st.file_uploader("Upload dataset image", type=["jpg", "png", "jpeg"])
        if image_input is not None:
            # Display the uploaded image.
            uploaded_image = Image.open(image_input).convert('RGB')
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            filename = image_input.name
            dataset_folder = find_image_dataset_folder(filename)
            if dataset_folder is None:
                st.error("Could not locate the uploaded image in the dataset (gaussian_filtered_images).")
            else:
                st.info(f"Image found in dataset folder: **{dataset_folder}**")
            # Run model prediction on the image.
            pred_class, confidence, prediction = dr_prediction(image_input)
            if pred_class is not None:
                st.success(f"Model Prediction: **{pred_class}** with {confidence:.2f}% confidence.")
                st.write("Raw prediction vector:", prediction[0])
                fig = plot_prediction_distribution(prediction)
                st.pyplot(fig)
            else:
                st.error("Prediction failed. Check debug output above.")
            # Finally, analyze the dataset folder (either from file lookup or prediction)
            class_to_analyze = dataset_folder if dataset_folder is not None else pred_class
            if class_to_analyze:
                analyze_dataset(class_to_analyze)
    
    elif choice == "Model Evaluation":
        display_model_evaluation()
    
    # Hide default Streamlit elements.
    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
