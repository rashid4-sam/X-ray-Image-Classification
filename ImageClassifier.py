import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title('X-Ray Image Classifier')
img_size = 100
PRETRAINED_MODEL_PATH = "custom_pre_trained_model_10.h5"
CATEGORIES = ["NORMAL", "PNEUMONIA"]

model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
print('Model Loaded')


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

    


    