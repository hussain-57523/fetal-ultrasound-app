# app.py

import streamlit as st
from PIL import Image
import torch

# Import your custom modules
from model.fetalnet import FetalNet
from utils.preprocess import transform_image
from predict import make_prediction

# Use caching to load the model only once
@st.cache_resource
def load_model():
    """
    Loads the trained FetalNet model from the saved .pth file.
    """
    # Instantiate the model architecture
    # Ensure num_classes_model matches your trained model
    model = FetalNet(num_classes_model=6) 
    
    # Path to your best model file
    model_path = "model/trained_models/best_model.pth"
    
    # Load the state dictionary
    # Use map_location to ensure it works on CPU if no GPU is available
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()
    return model

# --- Main App Interface ---

# Load the model
model = load_model()

# Set up the title and description
st.title("Fetal Ultrasound Plane Classifier 🔬")
st.write(
    "Upload a fetal ultrasound image, and the model will predict its anatomical plane. "
    "This model can identify Abdomen, Brain, Femur, Thorax, Maternal Cervix, or 'Other'."
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")

    # When the user clicks the button, make a prediction
    if st.button('Classify Plane'):
        with st.spinner('Analyzing the image...'):
            # 1. Preprocess the image
            tensor = transform_image(uploaded_file)
            
            if tensor is not None:
                # 2. Make a prediction
                predicted_class, confidence, all_probs = make_prediction(model, tensor)

                # 3. Display the result
                st.success(f"**Predicted Plane: {predicted_class}**")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Optional: Display all class probabilities
                st.write("---")
                st.write("**All Class Probabilities:**")
                for class_name, prob in all_probs.items():
                    st.write(f"{class_name}: {prob:.2%}")

            else:
                st.error("Could not process the uploaded image. Please try another one.")

# Add a sidebar with some information
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a FetalNet deep learning model (built with PyTorch) "
    "to classify fetal ultrasound images. The model architecture uses a "
    "pre-trained MobileNetV2 base with a custom head including a "
    "Squeeze-and-Excitation (SE) block."
)
st.sidebar.header("Example Images")
# st.sidebar.write("Try uploading one of the sample images from the `examples` folder in the GitHub repo.")