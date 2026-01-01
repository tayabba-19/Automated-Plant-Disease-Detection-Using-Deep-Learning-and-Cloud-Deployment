import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Automated Plant Disease Detection",
    layout="centered"
)

st.title("🌿 Automated Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease in real time.")

# -------------------------------
# Debug (optional – remove later)
# -------------------------------
# st.write("Files in directory:", os.listdir())

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_trained_model()

# -------------------------------
# Class Names (ORDER MUST MATCH TRAINING)
# ⚠️ Ye order wahi hona chahiye jisme model train hua
# -------------------------------
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Healthy"
]

# -------------------------------
# Suggested Actions
# -------------------------------
suggested_actions = {
    "Tomato___Early_blight": "Apply fungicide immediately and remove infected leaves.",
    "Tomato___Late_blight": "Use copper-based fungicide and avoid excess moisture.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato___Healthy": "No action required. Plant is healthy."
}

# -------------------------------
# Image Preprocessing (FOOLPROOF)
# -------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Show uploaded image
        original_img, img_array = preprocess_image(uploaded_file)
        st.image(original_img, caption="Uploaded Leaf Image", use_container_width=True)

        # Prediction
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions)) * 100
        class_index = int(np.argmax(predictions))
        disease_name = class_names[class_index]

        st.subheader("Prediction Result")

        # Confidence threshold (IMPORTANT)
        if confidence < 60:
            st.warning("Image is unclear. Please upload a clearer leaf image.")
        else:
            st.success(f"Disease: {disease_name}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.write(
                "**Suggested Action:**",
                suggested_actions.get(disease_name, "Consult an expert.")
            )

    except Exception as e:
        st.error("Error processing image.")
        st.write(e)
        

