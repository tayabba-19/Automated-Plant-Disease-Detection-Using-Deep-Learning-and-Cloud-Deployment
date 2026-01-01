import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Automated Plant Disease Detection",
    layout="centered"
)

st.title("🌱 Automated Plant Disease Detection")
st.write("Upload a tomato leaf image to detect plant disease.")

# ==============================
# Load Model (EXACT FILE NAME)
# ==============================
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_trained_model()

# ==============================
# Class Names (ORDER MUST MATCH TRAINING)
# ==============================
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Healthy"
]

# ==============================
# Suggested Actions
# ==============================
suggested_actions = {
    "Tomato___Early_blight": "Apply fungicide immediately.",
    "Tomato___Late_blight": "Remove infected leaves and apply fungicide.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Control whiteflies and isolate infected plants.",
    "Tomato___Healthy": "Plant is healthy. No action required."
}

# ==============================
# Image Preprocessing (FOOLPROOF)
# ==============================
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))   # same as training
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# ==============================
# File Uploader
# ==============================
uploaded_file = st.file_uploader(
    "Choose a leaf image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Preprocess image
        display_img, img_array = preprocess_image(uploaded_file)

        # Show image
        st.image(display_img, caption="Uploaded Leaf Image", use_container_width=True)

        # Predict
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions)) * 100
        class_index = int(np.argmax(predictions))
        disease_name = class_names[class_index]

        st.subheader("Prediction Result")

        # Confidence check
        if confidence < 60:
            st.warning("⚠️ Image unclear. Please upload a clear leaf image.")
        else:
            st.success(f"🌿 Disease: {disease_name}")
            st.info(f"📊 Confidence: {confidence:.2f}%")
            st.write(
                "**Suggested Action:**",
                suggested_actions.get(disease_name, "Consult an expert.")
            )

    except Exception as e:
        st.error("❌ Error processing image.")
        st.write(e)
        

