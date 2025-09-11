import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

st.title("🩻 X-Ray Image Classifier (Normal vs Pneumonia)")

# Model settings
IMG_SIZE = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="rashidsamad/pneumonia-detection",   
        filename="custom_pre_trained_model_10.h5"   
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# Classifier function
def load_classifier():
    st.subheader("Upload an X-Ray image to detect if it is Normal or Pneumonia")
    file = st.file_uploader("Upload X-Ray Image", type=['jpg', 'jpeg', 'png'])

    if file is not None:
        # Load and preprocess image
        img = Image.open(file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        st.image(file, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("🔍 Predict"):
            prediction = model.predict(img_array)
            prob = prediction[0][0]
            label = CATEGORIES[int(round(prob))]
            confidence = prob * 100 if label == "PNEUMONIA" else (1 - prob) * 100

            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

# Main app
def main():
    load_classifier()

if __name__ == "__main__":
    main()
