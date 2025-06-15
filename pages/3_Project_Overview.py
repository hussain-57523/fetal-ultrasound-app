import streamlit as st
import os
# Import the visualization function we created earlier
from utils.visualizations import plot_class_distribution

st.set_page_config(page_title="Project Overview", layout="wide")

st.title("📖 Project Overview")
st.markdown("""
Welcome to **FetalNet** – a deep learning system for classifying standard fetal ultrasound planes using a custom architecture and Explainable AI (XAI). This project demonstrates an end-to-end workflow from data preprocessing to final deployment.
""")

st.markdown("---")

# --- Project Objectives ---
st.header("🎯 Project Objectives")
st.markdown("""
- **Automate** the classification of fetal ultrasound images into key standard views to assist sonographers.
- **Design** a lightweight yet accurate custom CNN architecture (`FetalNet`).
- **Validate** the architectural design through systematic fine-tuning and ablation studies.
- **Build Trust** in the model’s predictions by implementing multiple Explainable AI techniques.
""")

st.markdown("---")

# --- Model Architecture ---
st.header("🏗️ FetalNet Architecture Summary")

# Show model architecture diagram
# Make sure you have this image saved in your 'assets' folder
arch_path = os.path.join("assets", "model_architecture_diagram.png")
if os.path.exists(arch_path):
    st.image(arch_path, caption="The FetalNet architecture combines a pre-trained base with a custom head.", use_column_width=True)
else:
    st.warning("Architecture diagram not found at 'assets/model_architecture_diagram.png'")

st.markdown("""
- **Base Model:** A pre-trained **MobileNetV2**, with its layers frozen to act as a powerful feature extractor.
- **Attention Mechanism:** A **Squeeze-and-Excitation (SE) Block** was integrated to help the model focus on the most relevant feature channels.
- **Classifier Head:** A custom 2-layer classifier with a `Dropout` layer for regularization makes the final prediction.
""")

st.markdown("---")

# --- Dataset Information ---
st.header("📦 Dataset: FETAL_PLANES_DB")
st.markdown("""
The model was trained on a publicly available dataset from Zenodo, containing thousands of labeled ultrasound images.

- **Data Split:** The dataset was divided into Training (70%), Validation (15%), and Testing (15%) sets.
- **Number of Classes:** 6
- **Class Labels:**
    - Fetal abdomen
    - Fetal brain
    - Fetal femur
    - Fetal thorax
    - Maternal cervix
    - Other
""")

# --- Display the Class Distribution Chart Dynamically ---
st.subheader("Class Distribution (Test Set)")
st.markdown("The chart below shows the number of images per class in the test set, highlighting the class imbalance.")
# This calls the function from your visualizations utility file
fig = plot_class_distribution()
st.pyplot(fig)


st.markdown("---")
# --- Performance Summary ---
st.header("🔍 Final Performance Summary")
st.success(
    "After a two-phase training process (initial training + fine-tuning), the final model achieved a **Test Accuracy of 93.01%**  "
    "and a **Weighted F1-Score of 0.93**. The ablation study further validated the importance of the final model's design."
)