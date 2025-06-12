# app.py
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from predict import load_model, preprocess_image
from utils.gradcam_utils import generate_gradcam
from model.custom_cnn import FetalNet

# --- Setup ---
st.set_page_config(page_title="Fetal Plane Classifier", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "model/trained_models/best_model.pth"
model = load_model(weights_path, device)

target_names = ["Abdomen", "Brain", "Femur", "Thorax", "Cervix", "Other"]
st.title("🩺 Fetal Ultrasound Plane Classifier + Grad-CAM")

uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(uploaded_file).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_label = target_names[pred_class]

    st.success(f"Predicted Class: **{pred_label}**")

    # Grad-CAM
    st.subheader("🔥 Grad-CAM Explanation")
    target_layer = model.conv[5]  # SE block is usually index 5
    cam_image = generate_gradcam(model, input_tensor, target_layer, pred_class, device=device.type)

    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
    st.write("This heatmap highlights the regions of the image that contributed most to the classification decision.")
    st.write("You can upload another image to see the classification and Grad-CAM results.")
else:
    st.info("Please upload an ultrasound image to classify and visualize with Grad-CAM.")
    st.write("Supported formats: PNG")
    st.write("The model will classify the image and provide a Grad-CAM visualization of the decision-making process.")