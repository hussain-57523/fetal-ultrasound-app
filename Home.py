# Home.py

import streamlit as st
from PIL import Image
import torch

# Import your custom modules
from model.fetalnet import FetalNet
from utils.preprocess import transform_image
from utils.prediction import make_prediction

# Use caching to load the model only once, improving performance
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned FetalNet model.
    """
    model = FetalNet(num_classes_model=6)
    # The path is relative to the root of your project folder
    model_path = "model/trained_models/fine_tuned_best_model.pth"
    # Load the state dictionary, mapping to CPU for broad compatibility
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- App Layout and Logic ---

# Load model
model = load_model()

# Set up the title and description
st.title("Fetal Ultrasound Plane Classifier 🔬")
st.markdown(
    "Welcome! This tool uses a deep learning model to classify fetal ultrasound images. "
    "Upload an image to see its predicted anatomical plane."
)

# File uploader
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("") # Add some space

    # When the user clicks the button, make a prediction
    if st.button('Classify Plane', type="primary"):
        with st.spinner('Analyzing the image...'):
            # 1. Preprocess the image and get the tensor
            tensor = transform_image(uploaded_file)
            
            if tensor is not None:
                # 2. Make a prediction
                predicted_class, confidence = make_prediction(model, tensor)

                # 3. Display the result
                st.success(f"**Predicted Plane: {predicted_class}**")
                st.write(f"**Confidence:**")
                st.progress(confidence, text=f"{confidence:.2%}")
                
                # Placeholder for Grad-CAM explanation
                st.info("Explainable AI (Grad-CAM) visualization can be added here to show model focus.")

            else:
                st.error("Could not process the uploaded image. Please try another one.")

# Add a sidebar with project information
st.sidebar.title("About the Project")
st.sidebar.info(
    "This app is the final deployment of a Final Year Project on Fetal Ultrasound Classification. "
    "The `FetalNet` model uses a pre-trained MobileNetV2 base with a custom head including a "
    "Squeeze-and-Excitation (SE) block, and was fine-tuned for high accuracy."
)
st.sidebar.markdown("---")
st.sidebar.write("Created by: **[Your Name Here]**")