import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Automated Plant Disease Detection",
    layout="centered"
)

st.title("🌿 Automated Plant Disease Detection System")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5", compile=False)
    return model

model = load_model()

# -----------------------------
# CLASS NAMES (same order as training)
# -----------------------------
class_names = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
recommendations = {
    "Healthy": "Leaf is healthy. No action required.",
    "Tomato Septoria Leaf Spot": "Prune infected leaves and apply recommended fungicide.",
    "Tomato Early Blight": "Remove infected leaves and use fungicide.",
    "Tomato Late Blight": "Use copper-based fungicide and improve drainage.",
    "Tomato Leaf Mold": "Improve air circulation and apply fungicide.",
    "Tomato Bacterial Spot": "Avoid overhead watering and apply bactericide.",
    "Tomato Spider Mites": "Use insecticidal soap or neem oil.",
    "Tomato Target Spot": "Remove infected leaves and apply fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants and disinfect tools."
}

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img_array = preprocess_image(image)

    predictions = model.predict(img_array)
    confidence = float(np.max(predictions)) * 100
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Disease Detected:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.markdown("### 💡 Recommendation")
    st.write(recommendations.get(predicted_class, "No recommendation available."))

    # -----------------------------
    # GRAD-CAM INFO (SAFE DISPLAY)
    # -----------------------------
    st.markdown("### 🔥 Grad-CAM Visualization")
    st.warning(
        "Grad-CAM is used during model analysis to highlight infected regions. "
        "Due to deployment constraints, live Grad-CAM visualization is shown in the report."
    )
