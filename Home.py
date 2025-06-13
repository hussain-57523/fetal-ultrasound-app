import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# Import your custom modules
from model.fetalnet import FetalNet
from utils.preprocess import transform_image
from utils.prediction import make_prediction
# Make sure your XAI utility file is named xai_techniques.py
from utils.xai_techniques import (
    generate_grad_cam, 
    generate_guided_backprop, 
    generate_integrated_gradients, 
    generate_occlusion_sensitivity
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use caching to load the model only once
@st.cache_resource
def load_model():
    model = FetalNet(num_classes_model=6).to(device)
    model_path = "model/trained_models/fine_tuned_best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- Main App Interface ---
st.set_page_config(layout="wide")
st.title("Fetal Ultrasound Plane Classifier with Explainable AI 🔬")
st.markdown(
    "Upload a fetal ultrasound image to classify its anatomical plane. After classification, "
    "explore the **Explainable AI (XAI)** tabs to understand *why* the model made its decision."
)

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    st.image(original_image, caption='Uploaded Image', use_column_width=True)
    st.write("") 

    if st.button('Classify Plane', type="primary", use_container_width=True):
        with st.spinner('Analyzing the image...'):
            tensor = transform_image(uploaded_file).to(device)
            
            if tensor is not None:
                predicted_class, confidence = make_prediction(model, tensor)

                st.success(f"**Predicted Plane: {predicted_class}**")
                st.write("**Confidence:**")
                st.progress(confidence, text=f"{confidence:.2%}")
                
                st.markdown("---")
                st.header("🤖 Explainable AI (XAI) Visualizations")
                st.write("These visualizations help us understand the model's decision-making process.")
                
                # --- XAI TABS SECTION (NOW WITH ERROR HANDLING) ---
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Grad-CAM", "Guided Backpropagation", 
                    "Integrated Gradients", "Occlusion Sensitivity"
                ])

                with tab1:
                    st.subheader("Grad-CAM: Where the Model Looks")
                    st.info("This heatmap shows the general **regions** the model found important. **Red areas** are the most influential.")
                    with st.spinner("Generating Grad-CAM..."):
                        grad_cam_viz, pred_name = generate_grad_cam(model, tensor, original_image)
                        if grad_cam_viz is not None:
                            st.image(grad_cam_viz, caption=f"Grad-CAM for prediction: '{pred_name}'", use_column_width=True)
                        else:
                            st.error("Could not generate Grad-CAM visualization. Check the logs for details.")

                with tab2:
                    st.subheader("Guided Backpropagation: What Details Matter")
                    st.info("This technique highlights the specific **pixels and edges** that had a positive influence on the final decision.")
                    with st.spinner("Generating Guided Backpropagation..."):
                        guided_bp_viz, pred_name = generate_guided_backprop(model, device, tensor)
                        if guided_bp_viz is not None:
                            st.image(guided_bp_viz, caption=f"Guided Backprop for prediction: '{pred_name}'", use_column_width=True)
                        else:
                            st.error("Could not generate Guided Backpropagation.")
                        
                with tab3:
                    st.subheader("Integrated Gradients: Stable Feature Importance")
                    st.info("This shows important pixels but is often **cleaner and less noisy** than other methods. It's great for identifying subtle features.")
                    with st.spinner("Generating Integrated Gradients..."):
                        ig_viz, pred_name = generate_integrated_gradients(model, tensor)
                        if ig_viz is not None:
                            fig, ax = plt.subplots()
                            ax.imshow(ig_viz, cmap='inferno')
                            ax.set_title(f"Integrated Gradients for '{pred_name}'")
                            ax.axis('off')
                            st.pyplot(fig)
                        else:
                            st.error("Could not generate Integrated Gradients.")

                with tab4:
                    st.subheader("Occlusion Sensitivity: Critical Region Test")
                    st.info("This heatmap shows which regions are most critical. **Bright areas** mean that hiding this part of the image would significantly confuse the model. **Note: This technique is very slow.**")
                    with st.spinner("Generating Occlusion Map (this can take up to a minute)..."):
                        occlusion_viz, pred_name = generate_occlusion_sensitivity(model, device, tensor)
                        if occlusion_viz is not None:
                            fig, ax = plt.subplots()
                            ax.imshow(original_image.resize((224, 224)), cmap='gray')
                            ax.imshow(occlusion_viz, cmap='jet', alpha=0.5)
                            ax.set_title(f"Occlusion Sensitivity for '{pred_name}'")
                            ax.axis('off')
                            st.pyplot(fig)
                        else:
                            st.error("Could not generate Occlusion Sensitivity map.")
            else:
                st.error("Could not process the uploaded image.")

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.info(
    "This app demonstrates a complete end-to-end deep learning project for fetal ultrasound classification, "
    "including multiple Explainable AI techniques to ensure model transparency and trustworthiness."
)