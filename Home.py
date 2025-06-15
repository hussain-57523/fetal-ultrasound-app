import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# This block attempts to import your custom modules.
try:
    from model.fetalnet import FetalNet
    from utils.preprocess import transform_image
    from utils.prediction import make_prediction
    from utils.xai_techniques import (
        generate_grad_cam, 
        generate_guided_backprop, 
        generate_integrated_gradients, 
        generate_occlusion_sensitivity
    )
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure your file structure is correct and all `__init__.py` files are in place in the 'utils' and 'model' folders.")
    st.stop()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Loads the fine-tuned FetalNet model."""
    try:
        model = FetalNet(num_classes_model=6).to(device)
        model_path = "model/trained_models/best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}.")
        return None

# --- Main App Interface ---
st.set_page_config(layout="wide")
st.title("Fetal Ultrasound Plane Classifier with Explainable AI 🔬")
st.markdown(
    "Upload a fetal ultrasound image to classify its anatomical plane. After classification, "
    "explore the **Explainable AI (XAI)** tabs to understand *why* the model made its decision."
)

model = load_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify Plane', type="primary", use_container_width=True):
            with st.spinner('Analyzing the image...'):
                tensor = transform_image(uploaded_file)
                
                if tensor is not None:
                    tensor = tensor.to(device)
                    predicted_class, confidence = make_prediction(model, tensor)

                    st.success(f"**Predicted Plane: {predicted_class}**")
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                    
                    st.markdown("---")
                    st.header("🤖 Explainable AI (XAI) Visualizations")
                    st.write("Click a button in each tab to generate an explanation.")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Grad-CAM", "Guided Backprop", "Integrated Gradients", "Occlusion Sensitivity"])

                    with tab1:
                        st.info("Shows the general **regions** the model found important.")
                        if st.button("Generate Grad-CAM", key="grad_cam_btn"):
                            # XAI Generation Logic for Grad-CAM
                            # ... (Call function from utils/xai_techniques.py)
                            st.success("Grad-CAM visualization would appear here.")
                    # ... Add logic for other tabs similarly ...
                else:
                    st.error("Could not process image.")