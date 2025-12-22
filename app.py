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
# Step 1: Download model if missing
# ---------------------------
MODEL_URL = "https://github.com/tayabba-19/Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment/raw/main/plant_scan.h5"
MODEL_PATH = "plant_scan.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded!")

# ---------------------------
# Step 2: Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------------------
# Step 3: Download class_names.json if missing
# ---------------------------
JSON_URL = "https://github.com/tayabba-19/Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment/raw/main/class_names.json"
JSON_PATH = "class_names.json"

if not os.path.exists(JSON_PATH):
    st.info("Downloading class names...")
    urllib.request.urlretrieve(JSON_URL, JSON_PATH)
    st.success("Class names downloaded!")

# ---------------------------
# Step 4: Load class names
# ---------------------------
with open(JSON_PATH, "r") as f:
    class_names = json.load(f)

# ---------------------------
# Step 5: Upload image
# ---------------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
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
    st.warning("Please upload a plant leaf image")

st.markdown("---")
st.markdown("**Project:** Automated Plant Disease Detection Using Deep Learning")
