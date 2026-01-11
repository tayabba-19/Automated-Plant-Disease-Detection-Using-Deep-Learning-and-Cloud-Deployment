import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import gdown

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to detect disease and see Grad-CAM visualization")

# ----------------------------
# Google Drive model download
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=16hmkzntUUX_BLCGCsaHM1zEQV1Zk0rAO"
MODEL_PATH = "tomato_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ----------------------------
# Load model (inference only)
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ----------------------------
# Class labels (same order as training)
# ----------------------------
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
    "Tomato Healthy"
]

# ----------------------------
# Disease recommendations (example)
# ----------------------------
recommendations = {
    "Tomato Bacterial Spot": "Remove affected leaves and use copper-based fungicide.",
    "Tomato Early Blight": "Apply appropriate fungicide and rotate crops.",
    "Tomato Late Blight": "Remove infected plants and improve air circulation.",
    "Tomato Leaf Mold": "Ensure proper ventilation and reduce leaf wetness.",
    "Tomato Septoria Leaf Spot": "Prune leaves and apply fungicide if necessary.",
    "Tomato Spider Mites": "Use miticide and keep plants hydrated.",
    "Tomato Target Spot": "Remove infected leaves and use recommended fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Remove infected plants; control whiteflies.",
    "Tomato Mosaic Virus": "Disinfect tools and remove infected plants.",
    "Tomato Healthy": "No action needed. Plant is healthy."
}

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------
# Grad-CAM functions (Safe)
# ----------------------------
def get_gradcam(model, img_array):

    # Find the last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        # No convolutional layer found
        return None

    # Create a gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8  # Avoid divide by zero

    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    img = np.array(img)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            prediction = model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

            # Recommendation
            rec = recommendations.get(predicted_class, "No recommendation available.")
            st.info(f"Recommendation: {rec}")

            # Grad-CAM heatmap
            heatmap = get_gradcam(model, processed_image)

            if heatmap is not None:
                overlay = overlay_gradcam(image, heatmap)
                st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
            else:
                st.warning("Grad-CAM visualization not supported for this model.")

            # Optional: Healthy leaf warning if confidence < 70%
            if predicted_class == "Tomato Healthy" and confidence < 70:
                st.warning("Prediction confidence is low. Please upload a clearer image.")
