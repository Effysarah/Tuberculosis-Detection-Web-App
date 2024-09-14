import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Load the trained model
model = load_model('cnn1.keras')

# Define image preprocessing function
def preprocess_image(image):
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to display confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About Me", "Dashboard"])

# Home page (Image Upload & Prediction)
if page == "Home":
    st.title("Tuberculosis Detection App")
    st.write("Upload a chest X-ray image to predict whether it is Normal or Tuberculosis.")

    # File uploader for chest X-ray image
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

# About Me page
elif page == "About Me":
    st.title("About Me")
     # Add your professional photo
    st.image("Photo.jpg", caption="Sarah Effiong", width=200)  # Replace "photo.jpg" with your actual file name

    st.write("""
    ### Developer: Sarah Effiong
    I am a Data Scientist, specializing in machine learning and deep learning applications. 
    This application uses a convolutional neural network (CNN) to assist in the early detection of 
    tuberculosis using chest X-rays.
    
    Feel free to reach out for any inquiries or collaborations!
    """)

   
 # Create columns for social media logos
    col1, col2, col3 = st.columns(3)

    with col1:
        # LinkedIn logo and link
        linkedin_logo = "Linkedln.png"  # Add the correct file path for the LinkedIn logo
        st.image(linkedin_logo, width=40)
        st.markdown("[![LinkedIn](https://image-link-placeholder)](https://www.linkedin.com/in/sarah-effiong-09913a210/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")  # Replace with your actual LinkedIn URL

    with col2:
        # GitHub logo and link
        github_logo = "github.png"  # Add the correct file path for the GitHub logo
        st.image(github_logo, width=40)
        st.markdown("[![GitHub](https://image-link-placeholder)](https://github.com/Effysarah)")  # Replace with your actual GitHub URL

    with col3:
        # Email logo and link
        email_logo = "email.png"  # Add the correct file path for the Email logo
        st.image(email_logo, width=40)
        st.markdown("[![Email](https://image-link-placeholder)](mailto:effysarah3108@gmail.com)")  # Replace with your actual email address


# Dashboard (ROC Curve, Confusion Matrix, Metrics)
elif page == "Dashboard":
    st.title("Model Performance Dashboard")
    
    # Model Performance Metrics (based on Test Set)
    st.write("### Model Performance (Test Set)")
    accuracy = 0.931116
    auc_value = 0.970370
    f1_score = 0.926616
    recall = 0.931116
    precision = 0.929619

    # Display the performance metrics
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC: {auc_value:.4f}")
    st.write(f"F1 Score: {f1_score:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"Precision: {precision:.4f}")

    # ROC Curve (example with placeholders, update with real test data)
    y_true = [0, 0, 1, 1, 0, 1]  # Ground truth labels (example)
    y_pred_proba = [0.1, 0.4, 0.8, 0.6, 0.3, 0.9]  # Model predicted probabilities (example)

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")

    # Display the ROC curve in Streamlit
    st.pyplot(fig)

    # Confusion Matrix
    y_pred = [0, 0, 1, 1, 0, 1]  # Predicted labels (example)
    plot_confusion_matrix(y_true, y_pred)

    # Conclusion
    st.write("### Conclusion:")
    st.write("The CNN model performs well on the test set with an accuracy of 93.11%, "
             "an AUC score of 97.04%, and a high F1 score of 92.66%. The precision (92.96%) "
             "and recall (93.11%) metrics also indicate a strong ability to correctly identify "
             "Tuberculosis cases while maintaining a low false positive rate.")
    

