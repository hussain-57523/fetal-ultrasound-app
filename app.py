import streamlit as st
from PIL import Image
import torch

# --- Import all custom modules ---
try:
    from model.fetalnet import FetalNet
    from utils.preprocess import preprocess_image
    from utils.prediction import make_prediction
    from utils.xai_techniques import generate_gradcam_figure
except ImportError as e:
    st.error(f"Import Error: {e}. Please check file structure and __init__.py files.")
    st.stop()

# --- Cached Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = FetalNet(num_classes_model=6).to(device)
    model_path = "model/trained_models/best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'.")
        return None, None

# --- Main App Interface ---
st.set_page_config(page_title="Fetal Ultrasound Classifier", layout="centered")
st.title("Fetal Ultrasound Plane Classifier 🔬")

model, device = load_model()

if model:
    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify and Explain", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                input_tensor = preprocess_image(image)
                if input_tensor is not None:
                    input_tensor = input_tensor.to(device)
                    
                    predicted_class, confidence, all_probs = make_prediction(model, input_tensor)
                    
                    st.success(f"**Predicted Plane:** {predicted_class}")
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                    
                    with st.expander("Show Detailed Probabilities"):
                        st.dataframe(all_probs.items(), column_config={"0": "Class", "1": "Confidence"}, hide_index=True)
                        
                    with st.expander("Show Model Explanation (XAI)"):
                        st.info("The heatmap below shows the regions the model focused on to make its decision. Brighter areas were more important.")
                        gradcam_fig = generate_gradcam_figure(model, input_tensor, image)
                        st.pyplot(gradcam_fig, use_container_width=True)
                else:
                    st.error("Could not preprocess the image.")
else:
    st.error("Model could not be loaded. The application cannot run.")