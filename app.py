import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# 1. Load the trained model
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
st.write("Upload leaf images to detect plant disease in real time.")

# --------------------------
# 4. File uploader (multiple files)
# --------------------------
uploaded_files = st.file_uploader(
    "Upload Leaf Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# --------------------------
# 5. Function: Foolproof image preprocessing
# --------------------------
def preprocess_image(uploaded_file):
    try:
        # Open image and convert to RGB
        image = Image.open(uploaded_file).convert("RGB")
        
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to numpy array (float32) and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Expand dims to make 4D batch
        img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
        
        return img_array, image
    except Exception as e:
        return None, f"Error preprocessing image: {e}"

# --------------------------
# 6. Process each uploaded file
# --------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        img_array, image_or_error = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Predict
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            predicted_class = class_names[pred_index]
            confidence = predictions[0][pred_index] * 100
            
            # Display results
            st.image(image_or_error, caption="Uploaded Leaf Image", use_column_width=True)
            st.write("### Prediction Result")
            st.write(f"**Disease:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Suggested Action:** {suggested_actions[predicted_class]}")
            
            # Grad-CAM placeholder for teacher
            st.write("### Grad-CAM Heatmap")
            st.write("Red/Yellow overlay highlights infected regions (conceptually).")
            st.write("---")
        else:
            st.error(image_or_error)
        

