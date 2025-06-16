import streamlit as st
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt

# --- 1. Import all custom modules with robust error checking ---
try:
    from model.fetalnet import FetalNet
    from utils.preprocess import transform_image
    from utils.prediction import make_prediction
    from utils.xai_techniques import generate_grad_cam, generate_guided_backprop
except ImportError as e:
    st.error(f"""
    **Import Error:** {e}
    
    This means a required file is missing or contains an error. Please ensure:
    1. The file structure is correct.
    2. All `__init__.py` files are present in the `model/` and `utils/` folders.
    3. All utility files (`fetalnet.py`, `preprocess.py`, etc.) are free of syntax errors.
    """)
    st.stop()

# --- 2. Define device and cached model loading function ---
device = torch.device("cpu")

@st.cache_resource
def load_model():
    """Loads the fine-tuned FetalNet model."""
    model = FetalNet(num_classes_model=6).to(device)
    model_path = "model/trained_models/best_model.pth"
    try:
        # Use weights_only=False for compatibility with older PyTorch save files
        # This directly addresses the UnpicklingError
        loaded_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle both full checkpoints and simple state_dict files
        if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
            model.load_state_dict(loaded_data['model_state_dict'])
        else:
            model.load_state_dict(loaded_data)
            
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"**Model file not found!** Please ensure `model/trained_models/best_model.pth` exists in your repository and that Git LFS is configured correctly.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- 3. Main App Interface ---
st.set_page_config(layout="wide", page_title="Fetal Ultrasound Classifier")
st.title("🔬 Fetal Ultrasound Plane Classifier with XAI")

model = load_model()

if model:
    uploaded_file = st.file_uploader("Upload a fetal ultrasound image to begin...", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Classify and Explain", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                input_tensor = transform_image(original_image)
                if input_tensor is not None:
                    input_tensor = input_tensor.to(device)
                    
                    predicted_class, confidence, _ = make_prediction(model, input_tensor)
                    
                    st.success(f"**Predicted Plane:** {predicted_class}")
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")

                    with st.expander("Show Explainable AI (XAI) Visualizations"):
                        viz_grad_cam, _ = generate_grad_cam(model, input_tensor, original_image)
                        viz_guided_bp, _ = generate_guided_backprop(model, device, input_tensor)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Grad-CAM")
                            st.info("Shows the general regions the model focused on.")
                            if viz_grad_cam is not None:
                                st.image(viz_grad_cam, use_column_width=True)
                            else:
                                st.error("Grad-CAM failed.")
                        with col2:
                            st.subheader("Guided Backprop")
                            st.info("Highlights the specific pixels that were most important.")
                            if viz_guided_bp is not None:
                                st.image(viz_guided_bp, use_column_width=True)
                            else:
                                st.error("Guided Backprop failed.")
                else:
                    st.error("Image could not be preprocessed.")
else:
    st.error("Model could not be loaded. Application cannot start.")