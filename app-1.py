import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('cnn1.keras')

# Define image preprocessing function
def preprocess_image(image):
    image = image.convert('RGB')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Tuberculosis Detection App")
st.write("Upload a chest X-ray image to predict whether it is Normal or Tuberculosis.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)

    # Preprocess the image
    prepared_image = preprocess_image(image)

    # Print the shape for debugging
    st.write(f"Prepared image shape: {prepared_image.shape}")

    # Make a prediction
    prediction = model.predict(prepared_image)
    probability = prediction[0][0]

    # Display the probability
    st.write(f"Prediction probability: {probability:.2f}")

    # Convert the prediction to a readable format
    if probability > 0.5:
        st.write("The model predicts: Tuberculosis")
    else:
        st.write("The model predicts: Normal")

    

