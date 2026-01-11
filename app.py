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
# Image preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------
# Grad-CAM functions
# ----------------------------
def get_gradcam(model, img_array, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    return heatmap

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

            # Grad-CAM heatmap
            heatmap = get_gradcam(model, processed_image)
            overlay = overlay_gradcam(image, heatmap)
            st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
