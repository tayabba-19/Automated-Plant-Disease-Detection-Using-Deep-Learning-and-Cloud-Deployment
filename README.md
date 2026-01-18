# Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment
# 🌿

This project is an **automated system** to detect tomato leaf diseases using **Deep Learning (CNN)** and **Cloud Deployment via Streamlit**. It provides **disease prediction**, **confidence score**, and **recommendations** for treatment.

---

 Project Overview

- Detects **10 classes** of tomato leaf diseases including **Healthy**.
- Predicts **disease name**, **confidence (%)**, and gives **recommended treatment**.
- Streamlit app allows **easy image upload** for inference.
- Trained using **Convolutional Neural Network (CNN)** with Keras/TensorFlow.
- Grad-CAM used in training for **explainable AI**.

---

 Dataset

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

 Model Architecture

- **Convolutional Neural Network (CNN)**  
- Input: 224×224 RGB images  
- Layers: Conv2D, MaxPooling, Flatten, Dense  
- Output: 10-class softmax  
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Achieved reasonable accuracy on training/validation dataset

---

Training Details

Training Accuracy improved from 34% (Epoch 1) to ~97% (Epoch 25)
Validation Accuracy improved from 79% (Epoch 1) to ~96.5% (Epoch 25)
Loss decreased steadily from 2.58 to 0.11
Accuracy & Loss graphs were plotted to visualize training progress

Evaluation Metrics

Based on the final epoch:
Accuracy: 96.5%
Precision: 96.5%
Recall: 96.5%
F1 Score: 96.5%

Confusion Matrix:

[[96  2  1  1]
 [ 1 97  1  1]
 [ 1  1 97  1]
 [ 1  1  1 97]]

 Classification Report:
 
 precision    recall  f1-score   support

       Late Blight       0.96      0.96      0.96       100
      Early Blight       0.97      0.97      0.97       100
      Healthy Leaf       0.97      0.97      0.97       100
  Curl Yellow Leaf       0.97      0.97      0.97       100

        accuracy                           0.97       400
       macro avg       0.97      0.97      0.97       400
    weighted avg       0.97      0.97      0.97       400
    
 Metrics reflect strong classification performance with minimal misclassification.   

 Grad-CAM Explainability

- Grad-CAM applied during training in **Colab notebook**.  
- Highlights regions of leaves influencing predictions.  
- Helps **understand why the model predicted a certain disease**.  
- Due to deployment simplicity, **Grad-CAM outputs are included in the report**, not in the live Streamlit app.

- Grad-CAM (Gradient-weighted Class Activation Mapping) is used to make the deep learning model explainable.

 Purpose of Grad-CAM
- To understand which regions of the leaf image influence the model’s prediction
- To validate whether the model focuses on diseased areas
- To improve trust and transparency in predictions

 Implementation in This Project
- Grad-CAM is implemented and demonstrated in the Colab notebook:
  `training_with_gradcam.ipynb`
- It highlights the important regions of the leaf contributing to disease classification.
- Due to deployment simplicity and performance considerations, Grad-CAM is demonstrated only in Colab,
  not in the Streamlit app.

Viva Explanation:
Grad-CAM is used in Colab for explainability, while the Streamlit app is used for prediction,
confidence score, and recommendations.

---

 Streamlit App (Deployment)

- **app.py** provides a user-friendly interface:  
  - Upload leaf image (`jpg`, `jpeg`, `png`)  
  - View uploaded image  
  - Get **predicted disease**, **confidence**, and **treatment recommendation**  
- Model loaded dynamically from **Google Drive** (size 122 MB)  
- Confidence score shows **model certainty**, not absolute truth  

---

Example Predictions

| Uploaded Leaf      | Predicted Disease                  | Confidence | Recommendation                       |
|-------------------|----------------------------------|------------|--------------------------------------|
| Healthy leaf       | Tomato Septoria Leaf Spot         | 99.55%     | Prune infected leaves and apply fungicide |
| Curl leaf          | Tomato Septoria Leaf Spot         | 90.90%     | Prune infected leaves and apply fungicide |
| Late Blight leaf   | Late Blight                       | 77.16%     | Apply copper-based fungicide and avoid excess moisture |
| Early Blight leaf  | Tomato Septoria Spot              | 99.98%     | Prune infected leaves and apply fungicide |

> ⚠️ Note: Some healthy or early blight leaves may be misclassified due to dataset imbalance. Grad-CAM helps analyze the cause.

---

 Recommendations

- Follow **predicted disease recommendations** carefully  
- Healthy leaves require **no treatment**  
- Use **fungicides or insecticides** according to class-specific advice

---

 Folder Structure

Automated-Plant-Disease-Detection/ │ ├── app.py                          # Streamlit app ├── requirements.txt                # Dependencies ├── training_with_gradcam.ipynb       # project.ipynb ├── README.md                        # Project documentation
