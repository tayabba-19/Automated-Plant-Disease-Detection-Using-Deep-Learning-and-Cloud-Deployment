import streamlit as st
import tensorflow as tf
import numpy as np
import gdown, os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ---------------- Page config ----------------
st.set_page_config(page_title="Tomato Leaf Disease Detection", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection System")

# ---------------- Google Drive model ----------------
MODEL_URL = "https://drive.google.com/uc?id=16hmkzntUUX_BLCGCsaHM1zEQV1Zk0rAO"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- Class labels ----------------
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

# ---------------- Recommendations ----------------
recommendations = {
    "Bacterial Spot": "Apply copper-based fungicide and avoid overhead irrigation.",
    "Early Blight": "Remove infected leaves and apply fungicide.",
    "Late Blight": "Use certified disease-free seeds and fungicide spray.",
    "Leaf Mold": "Improve air circulation and apply fungicide.",
    "Septoria Leaf Spot": "Remove infected leaves and avoid wet foliage.",
    "Spider Mites": "Use insecticidal soap or neem oil.",
    "Target Spot": "Use appropriate fungicide and crop rotation.",
    "Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Mosaic Virus": "Remove infected plants and disinfect tools.",
    "Healthy": "Plant is healthy. No action required."
}

# ---------------- Image preprocessing ----------------
def preprocess(image):
    image = image.resize((224,224))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- Grad-CAM ----------------
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload Tomato Leaf Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess(image)

    if st.button("Predict Disease"):
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        disease = class_names[index]
        confidence = prediction[0][index] * 100

        st.success(f"🌱 Disease Detected: **{disease}**")
        st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

        # Recommendation
        st.warning(f"📝 Recommended Action: {recommendations[disease]}")

        # Grad-CAM
        st.subheader("🔍 Disease Affected Area (Grad-CAM)")
        heatmap = grad_cam(model, img_array, model.layers[-5].name)

        img = cv2.resize(np.array(image), (224,224))
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.image(superimposed, use_column_width=True)
        

