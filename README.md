# 🐾 Cat vs Dog Classifier with SVM + MobileNetV2

A lightweight web app to classify images as **Cat or Dog**, and for dogs, it predicts the **breed** using deep features from MobileNetV2 and an SVM classifier.
Powered by Gradio for an interactive interface.

---

## 📊 Dataset

- Source: [Kaggle - Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
- Classes: 
  - Cats 🐱
  - Dogs 🐶
- Image Size: 224x224 resized

---

## 🧠 Method

- **Feature Extraction** using `MobileNetV2` (pretrained on ImageNet)
- **Classification** using `Support Vector Machine (SVM)`
- **Breed Prediction** using a lightweight deep learning model (MobileNetV2 based)

---

## 📈 Tools & Libraries

- Python
- TensorFlow / Keras
- OpenCV, NumPy
- scikit-learn
- Gradio (for UI)
- joblib (for saving/loading model)

---

## 📌 Steps

1. Load and preprocess dataset images
2. Extract features using MobileNetV2 (without top layers)
3. Train an SVM classifier on extracted features
4. Create an interactive web interface using Gradio
5. Deploy locally

---

## ✅ Output

- Predicts if the uploaded image is a **cat** or **dog**
- Displays **confidence score**
- Shows **dog breed** if predicted class is dog
- Sleek, user-friendly Gradio interface

---

## ▶️ Run Instructions

1. Clone the repo  
   `git clone (https://github.com/lavishsingh12/PRODIGY_ML_03.git)'

2. Run the app  
   `python main.py`

---

## 👨‍💻 Author

**Lavish Singh Rajawat**  
B.Tech Student | AI-ML Enthusiast  
GitHub: [@lavishsingh12](https://github.com/lavishsingh12)
