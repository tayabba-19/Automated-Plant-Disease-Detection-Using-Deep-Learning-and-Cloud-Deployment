import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Automated Plant Disease Detection",
    layout="centered"
)

st.title("🌿 Automated Plant Disease Detection")
st.write("Upload a tomato leaf image to detect disease in real time.")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

# ==============================
# Class Names
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
    "Tomato___Healthy": "No action required. Plant is healthy."
}

# ==============================
# Preprocess Image
# ==============================
def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    # Crop leaf: assume user uploads centered leaf
    width, height = img.size
    crop_size = min(width, height)
    left = (width - crop_size)//2
    top = (height - crop_size)//2
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))
    
    # Resize to model input
    img = img.resize((128, 128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# ==============================
# File Upload
# ==============================
uploaded_file = st.file_uploader(
    "Choose a leaf image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Preprocess
        display_img, img_array = preprocess_image(uploaded_file)
        
        # Show image
        st.image(display_img, caption="Uploaded Leaf Image", use_container_width=True)

        # Predict
        preds = model.predict(img_array)
        class_index = int(np.argmax(preds))
        disease_name = class_names[class_index]
        confidence = float(np.max(preds))*100

        st.subheader("Prediction Result")

        # Confidence check
        if confidence < 60:
            st.warning("⚠️ Image unclear. Please upload a clear leaf image.")
        else:
            # Display same name on top and bottom
            st.success(f"🌿 Disease: {disease_name}")
            st.info(f"📊 Confidence: {confidence:.2f}%")
            st.write("**Suggested Action:**", suggested_actions.get(disease_name, "Consult expert"))

    except Exception as e:
        st.error("❌ Error processing image. Upload a valid leaf image.")
        st.write(e)
        

