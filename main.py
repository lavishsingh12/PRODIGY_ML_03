import gradio as gr
import numpy as np
import cv2
import joblib
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

# Load models
svm_model = joblib.load("svm__model.pkl")
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
breed_model = MobileNetV2(weights="imagenet")


# Predict function
def predict(image):
    try:
        img = cv2.resize(image, (224, 224))
        processed_img = preprocess_input(img.copy())
        processed_img = np.expand_dims(processed_img, axis=0)

        # Extract features
        features = feature_extractor.predict(processed_img, verbose=0)
        features_flat = features.flatten().reshape(1, -1)

        # SVM Prediction
        label = svm_model.predict(features_flat)[0]
        proba = svm_model.predict_proba(features_flat)[0]

        # Breed detection for dogs
        breed_info = ""
        if label == 1:
            preds = breed_model.predict(processed_img, verbose=0)
            decoded = decode_predictions(preds, top=1)[0][0]
            breed_name = decoded[1].replace("_", " ").title()
            breed_info = f"<br><b style='color:#5A189A;'>Breed:</b> {breed_name}"

        if label == 0:
            result = f"""
            <div style='text-align:center; font-size: 24px; color: #EF476F; font-weight:bold;'>
                🐱 It's a Cat <br> Confidence: {proba[0]*100:.2f}%
            </div>
            """
        else:
            result = f"""
            <div style='text-align:center; font-size: 24px; color: #118AB2; font-weight:bold;'>
                🐶 It's a Dog <br> Confidence: {proba[1]*100:.2f}% {breed_info}
            </div>
            """
        return result, image
    except Exception as e:
        return f"<b>Error:</b> {str(e)}", None


def cancel():
    return "", None


# UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h1 style='text-align:center; color:#073B4C;'>🐾 Cat vs Dog Classifier</h1>"
    )

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload an Image", image_mode="RGB")
        image_preview = gr.Image(label="Uploaded Preview")

    output_label = gr.HTML()

    with gr.Row():
        submit_btn = gr.Button("🔍 Predict", variant="primary")
        cancel_btn = gr.Button("❌ Cancel", variant="secondary")

    submit_btn.click(
        fn=predict, inputs=image_input, outputs=[output_label, image_preview]
    )
    cancel_btn.click(fn=cancel, inputs=None, outputs=[output_label, image_preview])

demo.launch()
