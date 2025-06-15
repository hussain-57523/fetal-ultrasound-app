import streamlit as st
from PIL import Image
import torch
import os

# --- 1. Import from your custom utility files ---
# This structure assumes your utility functions are well-defined.
try:
    from model.fetalnet import FetalNet
    from utils.preprocess import transform_image
    from utils.prediction import make_prediction
    from utils.xai_techniques import generate_grad_cam # We will focus on Grad-CAM for the main UI
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure all utility and model files are in place and correct.")
    st.stop()

# --- 2. Function to load the model (cached for performance) ---
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned FetalNet model from the saved .pth file.
    This function runs only once.
    """
    # Set device to CPU for broad compatibility in deployment
    device = torch.device("cpu")
    
    # Instantiate the model architecture with the CORRECT number of classes
    model = FetalNet(num_classes_model=6).to(device)
    
    # Path to your best model file
    model_path = "model/trained_models/fine_tuned_best_model.pth"
    
    try:
        # Load the saved weights into the model structure
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set the model to evaluation mode
        print("✅ Model loaded successfully.")
        return model, device
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the model file is present in your repository.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# --- 3. Main App Interface ---
st.set_page_config(page_title="Fetal Ultrasound Classification", layout="wide")

st.title("🧠 Fetal Ultrasound Plane Classifier")
st.markdown("Upload a fetal ultrasound image to classify it into its anatomical plane and see an explanation of the model's decision.")

# Load the model when the app starts
model, device = load_model()

# Only proceed if the model was loaded successfully
if model is not None:
    # Use the main area for file upload for a better user flow
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        original_image = Image.open(uploaded_file).convert("RGB")   
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        # A primary button to trigger the analysis
        if st.button("🔍 Classify and Explain", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Preprocess the image to get a tensor
                input_tensor = transform_image(original_image)
                
                if input_tensor is not None:
                    input_tensor = input_tensor.to(device)
                    
                    # --- Prediction ---
                    predicted_class, confidence, _ = make_prediction(model, input_tensor)
                    st.success(f"**Predicted Plane:** {predicted_class}")
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                    st.markdown("---")

                    # --- Explainable AI Section ---
                    st.subheader("🤖 Explainable AI (XAI) Visualization")
                    with st.expander("See how the model made its decision"):
                        st.info("The heatmap below (Grad-CAM) highlights the regions in the image that were most important for the model's prediction. **Red areas** were the most influential.")
                        
                        # Generate the Grad-CAM visualization
                        gradcam_viz, _ = generate_grad_cam(model, input_tensor, original_image)
                        
                        if gradcam_viz is not None:
                            st.image(gradcam_viz, caption="Grad-CAM: Model's Focus Heatmap", use_column_width=True)
                        else:
                            st.error("Could not generate XAI visualization for this image.")
                else:
                    st.error("There was an error preprocessing the image.")
else:
    st.error("The model could not be loaded. The application cannot run.")