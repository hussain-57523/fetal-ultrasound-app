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
        model_path = "model/trained_models/fine_tuned_best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure the model exists in your repository and the path is correct.")
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
        st.write("") 

        if st.button('Classify Plane', type="primary", use_container_width=True):
            with st.spinner('Analyzing the image...'):
                tensor = transform_image(uploaded_file)
                
                if tensor is not None:
                    tensor = tensor.to(device)
                    predicted_class, confidence = make_prediction(model, tensor)

                    st.success(f"**Predicted Plane: {predicted_class}**")
                    st.write("**Confidence:**")
                    st.progress(confidence, text=f"{confidence:.2%}")
                    
                    st.markdown("---")
                    st.header("🤖 Explainable AI (XAI) Visualizations")
                    st.write("Click a button below to generate an explanation for the prediction.")
                    
                    # --- XAI TABS SECTION ---
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Grad-CAM", "Guided Backpropagation", 
                        "Integrated Gradients", "Occlusion Sensitivity"
                    ])

                    with tab1:
                        st.info("Shows the general **regions** the model found important. Red areas are most influential.")
                        if st.button("Generate Grad-CAM", key="grad_cam_btn"):
                            with st.spinner("Generating Grad-CAM..."):
                                viz, pred_name = generate_grad_cam(model, tensor, original_image)
                                if viz is not None:
                                    st.image(viz, caption=f"Grad-CAM for prediction: '{pred_name}'", use_column_width=True)
                                else:
                                    st.error("Could not generate Grad-CAM visualization.")

                    with tab2:
                        st.info("Highlights the specific **pixels and edges** that positively influenced the decision.")
                        if st.button("Generate Guided Backpropagation", key="gb_btn"):
                            with st.spinner("Generating Guided Backpropagation..."):
                                viz, pred_name = generate_guided_backprop(model, device, tensor)
                                if viz is not None:
                                    st.image(viz, caption=f"Guided Backprop for prediction: '{pred_name}'", use_column_width=True)
                                else:
                                    st.error("Could not generate Guided Backpropagation.")
                            
                    with tab3:
                        st.info("Shows important pixels but is often **cleaner and less noisy**. Good for subtle features.")
                        if st.button("Generate Integrated Gradients", key="ig_btn"):
                            with st.spinner("Generating Integrated Gradients..."):
                                viz, pred_name = generate_integrated_gradients(model, tensor)
                                if viz is not None:
                                    fig, ax = plt.subplots()
                                    ax.imshow(viz, cmap='inferno')
                                    ax.set_title(f"Integrated Gradients for '{pred_name}'")
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.error("Could not generate Integrated Gradients.")

                    with tab4:
                        st.info("Shows which regions are **critical** by hiding them. Bright areas are most important. **Note: This is very slow.**")
                        if st.button("Generate Occlusion Sensitivity (Slow)", key="occ_btn"):
                            with st.spinner("Generating Occlusion Map (this can take up to a minute)..."):
                                viz = generate_occlusion_sensitivity(model, device, tensor)
                                if viz is not None:
                                    fig, ax = plt.subplots()
                                    ax.imshow(original_image.resize((224, 224)), cmap='gray')
                                    ax.imshow(viz, cmap='jet', alpha=0.5)
                                    ax.set_title(f"Occlusion Sensitivity")
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.error("Could not generate Occlusion Sensitivity map.")
                else:
                    st.error("Could not process the uploaded image.")

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.info(
    "This app demonstrates an end-to-end deep learning project for fetal ultrasound classification, "
    "including multiple Explainable AI techniques to ensure model transparency and trustworthiness."
)