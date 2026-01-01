import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# 1. Model Load
# --------------------------
model = load_model("plant_disease_model.h5")

# --------------------------
# 2. Class Names
# --------------------------
class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Healthy"
]

# Suggested actions
suggested_actions = {
    "Tomato_Early_blight": "Apply fungicide immediately.",
    "Tomato_Late_blight": "Apply fungicide and remove infected leaves.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies; isolate infected plants.",
    "Tomato_Healthy": "No action needed."
}

# --------------------------
# 3. Streamlit UI
# --------------------------
st.title("🌱 Automated Plant Disease Detection")

st.write("Upload a leaf image to detect plant disease in real time.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --------------------------
    # 4. Image Preprocessing
    # --------------------------
    image = Image.open(uploaded_file).convert("RGB")  # convert to RGB
    image_resized = image.resize((224, 224))          # resize to model input
    img_array = np.array(image_resized) / 255.0       # normalize
    img_array = np.expand_dims(img_array, axis=0)     # add batch dimension

    # --------------------------
    # 5. Prediction
    # --------------------------
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    predicted_class = class_names[pred_index]
    confidence = predictions[0][pred_index] * 100

    # --------------------------
    # 6. Display Results
    # --------------------------
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    st.write("### Prediction Result")
    st.write(f"**Disease:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Suggested Action:** {suggested_actions[predicted_class]}")
    
    # --------------------------
    # 7. Optional: Grad-CAM Placeholder
    # --------------------------
    st.write("### Grad-CAM Heatmap")
    st.write("Red/Yellow overlay will highlight infected areas (conceptually).")
        

