import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Automated Plant Disease Detection",
    layout="centered"
)

st.title("🌱 Automated Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease in real time.")

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# =========================
# Class names (MUST match training order)
# =========================
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy"
]

# =========================
# Image preprocessing (FOOLPROOF)
# =========================
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# =========================
# File uploader
# =========================
uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Preprocess
        display_img, img_array = preprocess_image(uploaded_file)

        # Show uploaded image
        st.subheader("Uploaded Leaf Image")
        st.image(display_img, use_container_width=True)

        # Prediction
        preds = model.predict(img_array)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100
        disease_name = class_names[class_index]

        st.subheader("Prediction Result")

        # Confidence threshold
        if confidence < 60:
            st.warning("⚠️ Image unclear. Please upload a clear leaf image.")
        else:
            st.success(f"🌿 Disease: **{disease_name}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")

            # Suggested action
            if "Healthy" in disease_name:
                st.success("✅ Leaf is healthy. No action required.")
            elif "Early_blight" in disease_name:
                st.warning("🧪 Suggested Action: Apply appropriate fungicide.")
            elif "Late_blight" in disease_name:
                st.warning("🧪 Suggested Action: Remove infected leaves and apply fungicide.")
            elif "Yellow_Leaf_Curl" in disease_name:
                st.warning("🧪 Suggested Action: Control whiteflies and remove infected plants.")
            else:
                st.warning("🧪 Suggested Action: Consult agricultural expert.")

    except Exception as e:
        st.error("❌ Error processing image. Please upload a valid leaf image.")
        

