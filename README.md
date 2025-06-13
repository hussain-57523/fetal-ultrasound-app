# Fetal Ultrasound Plane Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[YOUR-STREAMLIT-APP-URL-HERE])  <-- *Add your app's link here after deployment!*

## Project Overview

This project is a deep learning application designed to automatically classify fetal ultrasound images into one of six key anatomical planes. The goal is to create a tool that can assist medical professionals by improving the efficiency and consistency of diagnoses. This repository contains the code for a Streamlit web application that provides a user-friendly interface for the trained model.

## Key Features

- **Live Prediction:** Upload a fetal ultrasound image and receive an instant classification.
- **Performance Dashboard:** An interactive, multi-page dashboard showcasing detailed model performance metrics.
- **Ablation Study Analysis:** Presents the results of a systematic study justifying the final model architecture.
- **Explainable AI (XAI):** *[Optional: Mention if you plan to add the Grad-CAM visualization to the app]*

## Model and Performance

- **Model:** The `FetalNet` architecture uses a frozen MobileNetV2 backbone with a custom head that includes a Squeeze-and-Excitation (SE) Block.
- **Final Performance:** The best model was achieved after a two-phase training process (initial training + fine-tuning).
  - [cite_start]**Final Test Accuracy:** **93.01%** 
  - [cite_start]**Final Weighted F1-Score:** **0.93** 

## How to Run Locally

This application can be run locally using a Conda environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hussain-Innovator/fetal-ultrasound-xai-app.git
    cd FetalNet_Streamlit_App
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name fyp_env python=3.9
    conda activate fyp_env
    ```
3.  **Install dependencies using Conda and Pip:**
    ```bash
    conda install pytorch torchvision cpuonly -c pytorch
    pip install streamlit pandas opencv-python-headless matplotlib seaborn grad-cam
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run 🏠_Home.py
    ```

## Project Structure

The repository is organized as follows:
- `/model`: Contains the `FetalNet` model architecture and the trained weights.
- `/pages`: Contains the scripts for each page of the Streamlit dashboard.
- `/utils`: Contains helper functions for preprocessing, prediction, and visualizations.
- `🏠_Home.py`: The main script for the Streamlit application.
- `requirements.txt`: The list of dependencies for Streamlit Cloud deployment.