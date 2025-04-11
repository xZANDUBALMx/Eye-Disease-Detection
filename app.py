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

# Define the base directory as the directory where app.py is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DR Diagnosis Mapping – keys must match your model's label encoding.
DIAGNOSIS_DICT = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# Cache the model loading so it doesn't reload every time.
@st.cache_resource
def load_dr_model():
    model_path = os.path.join(BASE_DIR, 'efficientnetb1.keras')
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_input):
    try:
        # Convert image to RGB and resize.
        size = (224, 224)
        image = Image.open(image_input).convert('RGB')
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.expand_dims(normalized_image_array, axis=0)
        return data
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def dr_prediction(image_input):
    data = preprocess_image(image_input)
    if data is None:
        return None, None, None
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
    # Create an "uploads" directory if it doesn't exist.
    uploads_dir = os.path.join(BASE_DIR, "uploads")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    # Create a unique folder with a timestamp.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = os.path.join(uploads_dir, time_stamp)
    os.makedirs(new_folder, exist_ok=True)
    # Keep the original extension.
    file_ext = os.path.splitext(uploaded_file.name)[1]
    save_path = os.path.join(new_folder, f"uploaded{file_ext}")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def analyze_dataset(pred_class):
    # Use the predicted class to point to the corresponding dataset folder.
    dataset_folder = os.path.join(BASE_DIR, "gaussian_filtered_images", pred_class)
    if not os.path.exists(dataset_folder):
        st.error(f"Dataset folder for class '{pred_class}' not found.")
        return
    
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        st.write("No images found in the predicted class folder.")
        return
    
    st.subheader("Sample Images from Predicted Class")
    sample_count = min(25, len(image_files))
    sampled_files = np.random.choice(image_files, sample_count, replace=False)
    cols = st.columns(5)
    for i, img_path in enumerate(sampled_files):
        with cols[i % 5]:
            st.image(img_path, use_column_width=True)
    
    # Calculate and plot prediction distribution in the folder.
    st.subheader("Prediction Distribution in Predicted Class Folder")
    model = load_dr_model()
    predictions_list = []
    num_images_to_predict = min(100, len(image_files))
    for img_path in image_files[:num_images_to_predict]:
        try:
            with Image.open(img_path).convert('RGB') as img:
                size = (224, 224)
                img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
                img_array = np.asarray(img)
                normalized_img = (img_array.astype(np.float32) / 127.0) - 1
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

def main():
    st.title("DetAll: Diabetic Retinopathy Detection")
    st.sidebar.title("Navigation")
    menu_options = ["Home", "DR Detection", "Details", "Model Evaluation", "Dataset Analysis"]
    choice = st.sidebar.selectbox("Menu", menu_options)

    if choice == "Home":
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text("Setting up the DR detection tool...")
            time.sleep(0.01)
        status_text.success("Ready!")
        st.write("This application detects diabetic retinopathy from eye images using a pre-trained EfficientNetB1 model.")
    
    elif choice == "DR Detection":
        st.sidebar.write("Upload an eye image for diabetic retinopathy detection.")
        image_input = st.sidebar.file_uploader("Choose an eye image", type=["jpg", "png"])
        if image_input is not None:
            # Save and display the uploaded image.
            saved_path = save_uploaded_file(image_input)
            st.write(f"Image successfully saved at: **{saved_path}**")
            saved_image = Image.open(saved_path).convert('RGB')
            display_size = st.slider("Adjust displayed image size:", 300, 1000, 500)
            st.image(saved_image, width=display_size, caption="Uploaded Image")
            # Reset the file pointer for prediction.
            image_input.seek(0)
            if st.sidebar.button("Analyze DR"):
                with st.spinner("Processing..."):
                    pred_class, confidence, prediction = dr_prediction(image_input)
                    if pred_class is not None:
                        st.success(f"Prediction: **{pred_class}** with {confidence:.2f}% confidence.")
                        st.write("Raw prediction vector:", prediction)
                        fig = plot_prediction_distribution(prediction)
                        st.pyplot(fig)
    
    elif choice == "Details":
        st.header("Detailed Visualizations")
        st.write("Below are graphs and visualizations showing model performance and training details.")
        details_folder = os.path.join(BASE_DIR, "details")
        if os.path.exists(details_folder):
            detail_images = sorted([f for f in os.listdir(details_folder)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if detail_images:
                for img_file in detail_images:
                    img_path = os.path.join(details_folder, img_file)
                    st.image(img_path, caption=img_file, use_column_width=True)
            else:
                st.write("No image files found in the 'details' folder.")
        else:
            st.write("The 'details' folder does not exist. Create a 'details' folder and add your visualizations there.")
    
    elif choice == "Model Evaluation":
        st.header("Model Evaluation Details")
        eval_file = os.path.join(BASE_DIR, "evaluation", "evaluation_data.npz")
        if os.path.exists(eval_file):
            data = np.load(eval_file)
            acc = data['acc']
            val_acc = data['val_acc']
            loss = data['loss']
            val_loss = data['val_loss']
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            st.subheader("Training History (Accuracy and Loss)")
            epochs = range(1, len(acc) + 1)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, acc, 'bo-', label="Training Accuracy")
            plt.plot(epochs, val_acc, 'ro-', label="Validation Accuracy")
            plt.title("Model Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(epochs, loss, 'bo-', label="Training Loss")
            plt.plot(epochs, val_loss, 'ro-', label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
            
            st.subheader("Confusion Matrix")
            class_names = [DIAGNOSIS_DICT[k] for k in sorted(DIAGNOSIS_DICT.keys())]
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            st.pyplot(plt)
            
            st.subheader("Classification Report")
            try:
                report = classification_report(y_true, y_pred, labels=[0,1,2,3,4], target_names=class_names)
                st.text(report)
            except Exception as e:
                st.error(f"Error generating classification report: {e}")
        else:
            st.write("Evaluation data file not found. Generate it using create_evaluation_data.py and save it in the 'evaluation' folder.")
    
    elif choice == "Dataset Analysis":
        st.header("Dataset Analysis")
        st.write("Upload an image from the dataset to analyze its predictions.")
        image_input = st.file_uploader("Upload image for dataset analysis", type=["jpg", "png"])
        if image_input is not None:
            uploaded_image = Image.open(image_input).convert('RGB')
            st.image(uploaded_image, caption="Uploaded Image for Analysis", use_column_width=True)
            image_input.seek(0)
            pred_class, confidence, prediction = dr_prediction(image_input)
            if pred_class is not None:
                st.success(f"Predicted Class: **{pred_class}** with {confidence:.2f}% confidence.")
                st.write("Raw prediction vector:", prediction)
                analyze_dataset(pred_class)
    
    # Hide default Streamlit UI elements.
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
