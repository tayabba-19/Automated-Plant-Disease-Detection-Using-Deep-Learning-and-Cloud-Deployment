import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import urllib.request

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="centered")

st.title("🌱 Automated Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using Deep Learning")

# ---------------------------
# Step 1: Download files if missing
# ---------------------------
MODEL_URL = "https://github.com/tayabba-19/Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment/raw/main/plant_scan.h5"
JSON_URL = "https://github.com/tayabba-19/Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment/raw/main/class_names.json"

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            st.info(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            st.success(f"{filename} downloaded!")
        except Exception as e:
            st.error(f"Failed to download {filename}: {e}")

download_file(MODEL_URL, "plant_scan.h5")
download_file(JSON_URL, "class_names.json")

# ---------------------------
# Step 2: Load model safely
# ---------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("plant_scan.h5")
        except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# ---------------------------
# Step 3: Load class names safely
# ---------------------------
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"Failed to load class names: {e}")
    class_names = {}

# ---------------------------
# Step 4: Upload image
# ---------------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None and class_names:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction Result")
    st.success(f"Disease: **{class_names[str(class_index)]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")
else:
    if not model or not class_names:
        st.warning("Model or class names not loaded yet. Please check logs.")
    else:
        st.warning("Please upload a plant leaf image.")

st.markdown("---")
st.markdown("**Project:** Automated Plant Disease Detection Using Deep Learning")

