import streamlit as st
from PIL import Image
import torch
import os

# --- Import your simplified custom modules ---
try:
    from model.fetalnet import FetalNet
    from utils.preprocess import transform_image
    from utils.prediction import make_prediction
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure your file structure is correct.")
    st.stop()

# --- Cached Model Loading ---
# In your app.py or Home.py file

@st.cache_resource
def load_model():
    """
    Loads the fine-tuned FetalNet model from the saved .pth file.
    """
    device = torch.device("cpu")
    model = FetalNet(num_classes_model=6).to(device)
    model_path = "model/trained_models/best_model.pth"
    
    try:
        # Load the file using the older method by setting weights_only=False
        # This is necessary for models saved with earlier PyTorch versions.
        # The 'weights_only' parameter was added in PyTorch 2.6
        loaded_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if the loaded data is a dictionary (from a checkpoint) or just the state_dict
        if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
            model.load_state_dict(loaded_data['model_state_dict'])
        else:
            # Assumes the file contains only the state_dict
            model.load_state_dict(loaded_data)
            
        model.eval()
        print("✅ Model loaded successfully.")
        return model, device
        
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# --- Main App Interface ---
st.set_page_config(page_title="Fetal Ultrasound Classifier", layout="centered")
st.title("Fetal Ultrasound Plane Classifier 🔬")
st.markdown("Upload a fetal ultrasound image to classify it into its anatomical plane.")

model, device = load_model()

if model:
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Classify", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                input_tensor = transform_image(image)
                
                if input_tensor is not None:
                    input_tensor = input_tensor.to(device)
                    
                    predicted_class, confidence, all_probs = make_prediction(model, input_tensor)
                    
                    st.success(f"**Predicted Plane:** {predicted_class}")
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")

                    # The XAI section has been removed for stability.
                    with st.expander("Show All Probabilities"):
                        st.dataframe(all_probs.items(), column_config={"0": "Class", "1": "Confidence"}, hide_index=True)
                else:
                    st.error("There was an error preprocessing the image.")
else:
    st.error("The model could not be loaded. The application cannot run.")