import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model("plant_disease_model.h5")

# Class names (same order as training)
class_names = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy"
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌱 Automated Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease in real time.")

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocessing
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    predicted_class = class_names[np.argmax(predictions)]

    # -------------------------------
    # Output
    # -------------------------------
    st.success(f"🌿 Disease Detected: {predicted_class}")
    st.info(f"🔍 Confidence Score: {confidence:.2f}%")

    # -------------------------------
    # Recommendation / Suggest Action
    # -------------------------------
    st.subheader("✅ Recommended Action")

    if predicted_class == "Tomato Early Blight":
        st.warning(
            "Apply appropriate fungicide and remove infected leaves to prevent spread."
        )

    elif predicted_class == "Tomato Late Blight":
        st.warning(
            "Use fungicide immediately and avoid excess moisture around plants."
        )

    elif predicted_class == "Tomato Yellow Leaf Curl Virus":
        st.warning(
            "Control whiteflies, remove infected plants, and use virus-free seeds."
        )

    elif predicted_class == "Healthy":
        st.success(
            "The plant is healthy. No treatment required. Maintain proper care."
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Deep Learning based Plant Disease Detection using Streamlit")
        

