import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Import your custom modules
from model.fetalnet import FetalNet
from utils.preprocess import transform_image
from utils.prediction import make_prediction
from xai_utils import (
    generate_grad_cam, 
    generate_guided_backprop, 
    generate_integrated_gradients, 
    generate_occlusion_sensitivity
)

# Use caching to load the model only once
@st.cache_resource
def load_model():
    model = FetalNet(num_classes_model=6)
    model_path = "model/trained_models/fine_tuned_best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        if st.button('Classify Plane', type="primary", use_container_width=True):
            with st.spinner('Analyzing the image...'):
                tensor = transform_image(uploaded_file)
                
                if tensor is not None:
                    predicted_class, confidence = make_prediction(model, tensor)

                    st.success(f"**Predicted Plane: {predicted_class}**")
                    st.write("**Confidence:**")
                    st.progress(confidence, text=f"{confidence:.2%}")
                    
                    st.markdown("---")
                    st.header("🤖 Explainable AI (XAI) Visualizations")
                    
                    # Create tabs for each XAI method
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Grad-CAM", "Guided Backpropagation", 
                        "Integrated Gradients", "Occlusion Sensitivity"
                    ])

                    with tab1:
                        st.subheader("Grad-CAM: Where the Model Looks")
                        st.info("Grad-CAM highlights the general regions (a 'heatmap') that were most important for the prediction. Red areas are the most influential.")
                        grad_cam_viz, _ = generate_grad_cam(model, tensor, original_image)
                        st.image(grad_cam_viz, caption="Grad-CAM Heatmap Overlay", use_column_width=True)

                    with tab2:
                        st.subheader("Guided Backpropagation: Which Pixels Matter")
                        st.info("This technique creates a high-resolution map highlighting the specific pixels that positively contributed to the decision.")
                        guided_bp_viz, _ = generate_guided_backprop(model, tensor)
                        st.image(guided_bp_viz, caption="Guided Backpropagation Saliency Map", use_column_width=True)
                        
                    with tab3:
                        st.subheader("Integrated Gradients: Stable Pixel Importance")
                        st.info("This provides a cleaner, less noisy version of pixel importance by comparing the image to a black baseline image.")
                        ig_viz, _ = generate_integrated_gradients(model, tensor)
                        st.image(ig_viz, caption="Integrated Gradients Attribution Map", use_column_width=True)

                    with tab4:
                        st.subheader("Occlusion Sensitivity: Impact of Hiding Regions")
                        st.info("This heatmap shows which regions are most critical. Red areas indicate that covering this part of the image would cause the prediction confidence to drop significantly. **(Note: This visualization can be slow to generate)**.")
                        with st.spinner("Generating Occlusion Map (this may take a moment)..."):
                            occlusion_viz, _ = generate_occlusion_sensitivity(model, tensor, original_image)
                            
                            # Create the plot for occlusion sensitivity
                            fig, ax = plt.subplots()
                            ax.imshow(original_image.resize((224, 224)))
                            ax.imshow(occlusion_viz, cmap='jet', alpha=0.5)
                            ax.axis('off')
                            st.pyplot(fig)
                else:
                    st.error("Could not process the uploaded image.")

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.info(
    "This app demonstrates a complete end-to-end deep learning project for fetal ultrasound classification, "
    "including multiple Explainable AI techniques to ensure model transparency and trustworthiness."
)