import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("plant_disease_model.h5")

# 🔥 Automatically get model input size
_, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = model.input_shape

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Healthy"
]

suggested_actions = {
    "Tomato_Early_blight": "Apply fungicide immediately.",
    "Tomato_Late_blight": "Apply fungicide and remove infected leaves.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies; isolate infected plants.",
    "Tomato_Healthy": "No action needed."
}

st.title("🌱 Automated Plant Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(file):
    image = Image.open(file)

    # 🔥 Handle grayscale images
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, image

if uploaded_file is not None:
    try:
        img_array, original_image = preprocess_image(uploaded_file)

        # 🔥 Debug info (teacher bhi impress 😄)
        st.write("Model Input Shape:", model.input_shape)
        st.write("Image Shape Sent:", img_array.shape)

        predictions = model.predict(img_array)
        index = np.argmax(predictions[0])

        st.image(original_image, caption="Uploaded Leaf", use_column_width=True)
        st.success(f"Disease: {class_names[index]}")
        st.info(f"Confidence: {predictions[0][index]*100:.2f}%")
        st.warning(f"Suggested Action: {suggested_actions[class_names[index]]}")

    except Exception as e:
        st.error("❌ Prediction Error")
        st.error(str(e))
        

