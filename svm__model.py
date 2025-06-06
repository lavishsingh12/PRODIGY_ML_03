import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Path to your dataset
DATASET_DIR = "cats_dogs_light/train"
IMG_SIZE = 224

# Load MobileNetV2 without top layer
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")


def load_data():
    X = []
    y = []
    for label, folder in enumerate(["cat", "dog"]):
        folder_path = os.path.join(DATASET_DIR, folder)
        for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}s"):
            try:
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = preprocess_input(img)
                features = feature_extractor.predict(
                    np.expand_dims(img, axis=0), verbose=0
                )
                X.append(features.flatten())
                y.append(label)
            except:
                pass  # skip unreadable files
    return np.array(X), np.array(y)


# Load and extract features
X, y = load_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(svm, "svm__model.pkl")
print("✅ SVM model saved as svm__model.pkl")
