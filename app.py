import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import make_gradcam_heatmap
import cv2

st.title("🌿 Automated Plant Disease Detection System")

model = load_model("plant_disease_model.h5")

class_names = list(recommendations.keys())

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((128,128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]*100
    disease = class_names[class_index]

    st.success(f"Detected Disease: {disease}")
    st.info(f"Confidence Score: {confidence:.2f}%")

    # Recommendation
    st.warning(f"Suggested Action: {recommendations[disease]}")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, "conv2d")
    heatmap = cv2.resize(heatmap, img.size)
    heatmap = np.uint8(255 * heatmap)

    st.subheader("Grad-CAM Heatmap")
    st.image(heatmap, use_column_width=True)
        

