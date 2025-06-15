# FetalNet – Fetal Ultrasound Plane Classification with Explainable AI

FetalNet is a custom deep learning model built to classify fetal ultrasound images into standard anatomical planes using a lightweight architecture. This final-year project includes advanced visualization, model interpretability, and web-based deployment using Streamlit.

---

## Project Highlights

- **Dataset**: [FETAL_PLANES_DB](https://zenodo.org/record/3904280)
- **Model**: Custom architecture using MobileNetV2 + SE Block
- **Accuracy**: Achieved 93.01% test accuracy after fine-tuning
- **Ablation Study**: Verified importance of SE Block and layers
- **XAI Support**: Grad-CAM, Guided Backprop, Integrated Gradients, Occlusion
- **Deployment**: Interactive web app built with Streamlit

---

## Demo – Streamlit App

![app_ui](assets/example_brain.png)  
*Upload a fetal ultrasound image to see the predicted class and visual explanation.*

---

## ⚙️ Features

| Feature                        | Description                                             |
|-------------------------------|---------------------------------------------------------|
| Classification             | 5-class prediction from ultrasound image                |
| Fine-tuned Model           | Lightweight and optimized for low-resource environments |
| XAI Visualizations         | Explain predictions using visual heatmaps               |
| Performance Dashboard      | Accuracy/loss graphs for training                       |
| Ablation Study             | Performance of model variants without key components    |

---

## How to Run (Locally)

```bash
git clone https://github.com/your-username/fetalnet-ultrasound
cd fetalnet-ultrasound
pip install -r requirements.txt
streamlit run app.py
