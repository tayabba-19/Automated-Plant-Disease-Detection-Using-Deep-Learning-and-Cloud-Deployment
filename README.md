# Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment
# 🌿

This project is an **automated system** to detect tomato leaf diseases using **Deep Learning (CNN)** and **Cloud Deployment via Streamlit**. It provides **disease prediction**, **confidence score**, and **recommendations** for treatment.

---

## 1. Project Overview

- Detects **10 classes** of tomato leaf diseases including **Healthy**.
- Predicts **disease name**, **confidence (%)**, and gives **recommended treatment**.
- Streamlit app allows **easy image upload** for inference.
- Trained using **Convolutional Neural Network (CNN)** with Keras/TensorFlow.
- Grad-CAM used in training for **explainable AI**.

---

## 2. Dataset

- Dataset sourced from **Kaggle** (Tomato Leaf Diseases).
- Images include:
  - Healthy leaves
  - Tomato Septoria Leaf Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Bacterial Spot
  - Spider Mites
  - Target Spot
  - Yellow Leaf Curl Virus
  - Mosaic Virus
- Preprocessing: resize to 224×224, normalize pixel values.

---

## 3. Model Architecture

- **Convolutional Neural Network (CNN)**  
- Input: 224×224 RGB images  
- Layers: Conv2D, MaxPooling, Flatten, Dense  
- Output: 10-class softmax  
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Achieved reasonable accuracy on training/validation dataset

---

## 4. Grad-CAM Explainability

- Grad-CAM applied during training in **Colab notebook**.  
- Highlights regions of leaves influencing predictions.  
- Helps **understand why the model predicted a certain disease**.  
- Due to deployment simplicity, **Grad-CAM outputs are included in the report**, not in the live Streamlit app.

---

## 5. Streamlit App (Deployment)

- **app.py** provides a user-friendly interface:  
  - Upload leaf image (`jpg`, `jpeg`, `png`)  
  - View uploaded image  
  - Get **predicted disease**, **confidence**, and **treatment recommendation**  
- Model loaded dynamically from **Google Drive** (size 122 MB)  
- Confidence score shows **model certainty**, not absolute truth  

---

### Example Predictions

| Uploaded Leaf      | Predicted Disease                  | Confidence | Recommendation                       |
|-------------------|----------------------------------|------------|--------------------------------------|
| Healthy leaf       | Tomato Septoria Leaf Spot         | 99.55%     | Prune infected leaves and apply fungicide |
| Curl leaf          | Tomato Septoria Leaf Spot         | 90.90%     | Prune infected leaves and apply fungicide |
| Late Blight leaf   | Late Blight                       | 77.16%     | Apply copper-based fungicide and avoid excess moisture |
| Early Blight leaf  | Tomato Septoria Spot              | 99.98%     | Prune infected leaves and apply fungicide |

> ⚠️ Note: Some healthy or early blight leaves may be misclassified due to dataset imbalance. Grad-CAM helps analyze the cause.

---

## 6. Recommendations

- Follow **predicted disease recommendations** carefully  
- Healthy leaves require **no treatment**  
- Use **fungicides or insecticides** according to class-specific advice

---

## 7. Folder Structure

Automated-Plant-Disease-Detection/ │ ├── app.py                          # Streamlit app ├── requirements.txt                # Dependencies ├── training_with_gradcam.ipynb     # Colab notebook with Grad-CAM ├── plant_disease_old.ipynb         # Old notebook (optional) ├── README.md                        # Project documentation
