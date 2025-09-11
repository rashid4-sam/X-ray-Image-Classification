import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

st.title('X-Ray Image Classifier')
img_size = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# Load the trained model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="https://huggingface.co/rashidsamad/pneumonia-detection/tree/main",  
        filename="custom_pre_trained_model_10.h5"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

def load_classifier():
    st.subheader("Upload an X-Ray image to detect if it is Normal or Pneumonia")
    file = st.file_uploader(label=" ", type = ['jpeg'])

    if file!= None:
        img=tf.keras.preprocessing.image.load_img(file, target_size=(img_size,img_size))
        new_array=tf.keras.preprocessing.image.img_to_array(img)
        new_array = new_array.reshape(-1,img_size,img_size,3)
        st.image(file)
        st.write("")
        st.write("")
            
        if st.button("PREDICT"):
            #Making prediction
            preds = "" 
            prediction=model.predict(new_array/255.0)
            print(prediction)
            print(round(prediction[0][0]))
            preds = CATEGORIES[int(round(prediction[0][0]))] + " - " +  str(round(prediction[0][0]*100,2)) + "%"
            st.write(preds)

def main():
    load_classifier()


if __name__ == "__main__":
	main()

    


    
