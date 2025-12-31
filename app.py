import streamlit as st
import subprocess
import sys

# --------------------------
# Runtime install heavy packages
# --------------------------
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.13.0", "opencv-python-headless"])

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
import cv2

# --------------------------
# TITLE
# --------------------------
st.title("🌿 Automated Plant Disease Detection")
st.markdown("Upload a leaf image to detect disease and highlight infected areas (Grad-CAM).")

# --------------------------
# LOAD MODEL
# --------------------------
model = load_model("plant_disease_model.h5")

# --------------------------
# DISEASE CLASSES & RECOMMENDATIONS
# --------------------------
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
           'Apple___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

recommendations = {
    'Apple___Apple_scab': "Apply sulfur-based fungicide.",
    'Apple___Black_rot': "Remove infected areas and use fungicide.",
    'Apple___Cedar_apple_rust': "Spray with protective fungicide.",
    'Apple___healthy': "No disease detected. Keep proper care.",
    'Potato___Early_blight': "Apply fungicide and ensure proper irrigation.",
    'Potato___Late_blight': "Use copper-based fungicide immediately.",
    'Potato___healthy': "No disease detected. Maintain proper care."
}

# --------------------------
# FILE UPLOADER
# --------------------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

# --------------------------
# GRAD-CAM FUNCTION
# --------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap.numpy()

# --------------------------
# MAIN LOGIC
# --------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((128,128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)*100
    disease = classes[class_index]

    st.subheader("Prediction Result")
    st.success(f"Disease: {disease}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Suggested Action: {recommendations[disease]}")

    # Grad-CAM Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d")
    heatmap = cv2.resize(heatmap, img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    st.subheader("Grad-CAM Heatmap")
    st.image(superimposed_img, use_column_width=True)
        

