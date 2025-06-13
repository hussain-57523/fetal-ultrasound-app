# pages/3_Project_Overview.py

import streamlit as st
from utils.visualizations import plot_class_distribution

st.set_page_config(page_title="Project Overview", layout="wide")

st.title("Project Overview")

st.markdown("---")

# --- Problem Statement ---
st.header("Problem Statement and Objective")
st.markdown("""
The accurate identification of standard anatomical planes in fetal ultrasound scans is a critical but time-consuming task for sonographers. It is essential for diagnosing fetal abnormalities and monitoring growth. Misidentification can lead to missed diagnoses or unnecessary follow-ups.

**The objective of this project was to develop and evaluate a deep learning model, `FetalNet`, capable of automatically and accurately classifying fetal ultrasound images into six key anatomical planes.** This tool aims to assist medical professionals by improving efficiency and consistency in diagnoses.
""")


st.markdown("---")

# --- Dataset Information ---
st.header("Dataset Overview")
st.markdown("""
The model was trained on the publicly available **FETAL_PLANES_DB** dataset. This dataset contains thousands of ultrasound images curated and labeled by medical experts. For this project, the data was split into training (70%), validation (15%), and testing (15%) sets.
""")
st.pyplot(plot_class_distribution())
st.info("The chart above shows the number of images for each of the 6 classes in the test set, highlighting a notable class imbalance that the model must handle.")


st.markdown("---")

# --- Model Architecture ---
st.header("FetalNet Model Architecture")
st.markdown("""
The core of this project is the `FetalNet` model, a custom deep learning architecture designed for this specific task.
1.  **Backbone (Feature Extractor):** It uses a **MobileNetV2** model, pre-trained on the ImageNet dataset, as its backbone. The layers of this backbone were **frozen**, allowing us to leverage powerful, general-purpose features without the need for extensive training.
2.  **Custom CNN Head:** On top of the backbone, I added two custom convolutional stages to refine the features specifically for the ultrasound domain.
3.  **Squeeze-and-Excitation (SE) Block:** An attention mechanism was included between the custom convolutional stages. The SE Block helps the model learn to focus on the most important channel-wise features, which is crucial for dealing with the noise and variability in ultrasound images.
4.  **Classifier:** A final classifier with a `Dropout` layer takes these refined features and makes the final prediction.

*(You could display a diagram of your architecture here if you have one)*
""")
# Example of how to show an image from the assets folder:
# st.image("assets/model_architecture_diagram.png", caption="FetalNet Architecture Diagram")


# st.markdown("---")
# st.header("Project Repository")
# st.info("The complete code, training notebooks, and project files are available on GitHub.")
# # st.markdown("[Visit the GitHub Repository](https://github.com/Hussain-Innovator/fetal-ultrasound-xai-app.git)", unsafe_allow_html=True) # <-- CHANGE THIS LINK