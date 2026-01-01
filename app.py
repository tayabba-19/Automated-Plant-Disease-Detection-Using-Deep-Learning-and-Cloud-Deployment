import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# 1. Load trained model
# --------------------------
model = load_model("plant_disease_model.h5")

# --------------------------
# 2. Class names & suggested actions
# --------------------------
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

# --------------------------
# 3. Streamlit UI
# --------------------------
st.title("🌱 Automated Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease in real time.")

# --------------------------
# 4. File uploader
# --------------------------
uploaded_files = st.file_uploader(
    "Upload Leaf Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# --------------------------
# 5. Process uploaded files
# --------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # 5a. Open image and preprocess
            image = Image.open(uploaded_file)
            image = image.convert("RGB")          # Force RGB
            image_resized = image.resize((224, 224))
            img_array = np.array(image_resized, dtype=np.float32) / 255.0

            # Ensure shape = (1,224,224,3)
            if img_array.ndim == 3:
                img_array = np.expand_dims(img_array, axis=0)

            # 5b. Predict
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            predicted_class = class_names[pred_index]
            confidence = predictions[0][pred_index] * 100

            # 5c. Display results
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
            st.write("### Prediction Result")
            st.write(f"**Disease:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Suggested Action:** {suggested_actions[predicted_class]}")

            # 5d. Grad-CAM placeholder
            st.write("### Grad-CAM Heatmap")
            st.write("Red/Yellow overlay highlights infected regions (conceptually).")
            st.write("---")

        except Exception as e:
            st.error(f"Error processing image: {e}")
        

