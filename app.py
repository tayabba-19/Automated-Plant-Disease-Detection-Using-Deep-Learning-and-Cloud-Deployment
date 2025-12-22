import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Title
st.title("🌱 Automated Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using Deep Learning")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_scan.h5")
    return model

model = load_model()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Image uploader
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))   # same size as training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Result
    st.subheader("Prediction Result")
    st.success(f"Disease: **{class_names[str(class_index)]}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

else:
    st.warning("Please upload a plant leaf image")

# Footer
st.markdown("---")
st.markdown("**Project:** Automated Plant Disease Detection Using Deep Learning")
