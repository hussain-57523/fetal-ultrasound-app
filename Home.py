import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# This block attempts to import your custom modules. An error here is a common issue.
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
    st.stop() # Stop the app if essential modules are missing

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

if model is None:
    st.warning("Model could not be loaded. The application cannot proceed.")
else:
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
                    st.write("Click on a tab below to generate an explanation.")
                    
                    # --- XAI TABS SECTION (WITH ROBUST ERROR HANDLING) ---
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Grad-CAM", "Guided Backpropagation", 
                        "Integrated Gradients", "Occlusion Sensitivity"
                    ])

                    with tab1:
                        st.info("This heatmap shows the general **regions** the model found important. **Red areas** are the most influential.")
                        if st.button("Generate Grad-CAM"):
                            with st.spinner("Generating Grad-CAM..."):
                                viz = generate_grad_cam(model, tensor, original_image)
                                if viz is not None:
                                    st.image(viz, caption=f"Grad-CAM for prediction: '{predicted_class}'", use_column_width=True)
                                else:
                                    st.error("Could not generate Grad-CAM. Check the app logs for details.")

                    with tab2:
                        st.info("This highlights the specific **pixels and edges** that had a positive influence on the final decision.")
                        if st.button("Generate Guided Backpropagation"):
                            with st.spinner("Generating Guided Backpropagation..."):
                                viz = generate_guided_backprop(model, device, tensor)
                                if viz is not None:
                                    st.image(viz, caption=f"Guided Backprop for prediction: '{predicted_class}'", use_column_width=True)
                                else:
                                    st.error("Could not generate Guided Backpropagation. Check the app logs.")
                            
                    with tab3:
                        st.info("This shows important pixels but is often **cleaner and less noisy**. It is great for identifying subtle features.")
                        if st.button("Generate Integrated Gradients"):
                            with st.spinner("Generating Integrated Gradients..."):
                                viz = generate_integrated_gradients(model, tensor)
                                if viz is not None:
                                    fig, ax = plt.subplots()
                                    ax.imshow(viz, cmap='inferno')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.error("Could not generate Integrated Gradients. Check the app logs.")

                    with tab4:
                        st.info("This heatmap shows which regions are **critical**. Hiding a red area would significantly confuse the model. **Note: This is very slow.**")
                        if st.button("Generate Occlusion Sensitivity (Slow)"):
                            with st.spinner("Generating Occlusion Map (this can take up to a minute)..."):
                                viz = generate_occlusion_sensitivity(model, device, tensor)
                                if viz is not None:
                                    fig, ax = plt.subplots()
                                    ax.imshow(original_image.resize((224, 224)), cmap='gray')
                                    ax.imshow(viz, cmap='jet', alpha=0.5)
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.error("Could not generate Occlusion Sensitivity map. Check the app logs.")
                else:
                    st.error("Could not process the uploaded image.")
# Sidebar
st.sidebar.title("About the Project")
st.sidebar.info(
    "This app demonstrates a complete end-to-end deep learning project for fetal ultrasound classification, "
    "including multiple Explainable AI techniques to ensure model transparency and trustworthiness."
)