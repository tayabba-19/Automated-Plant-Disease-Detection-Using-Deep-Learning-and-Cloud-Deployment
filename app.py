import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --------------------------
# TITLE
# --------------------------
st.title("🌿 Automated Plant Disease Detection")
st.markdown("Upload a leaf image and get prediction with confidence and suggested action.")

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

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    # --------------------------
    # PREPROCESS IMAGE
    # --------------------------
    img_resized = img.resize((128,128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    # --------------------------
    # PREDICTION
    # --------------------------
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)*100
    disease = classes[class_index]

    # --------------------------
    # DISPLAY RESULTS
    # --------------------------
    st.subheader("Prediction Result")
    st.success(f"Disease: {disease}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Suggested Action: {recommendations[disease]}")
        

