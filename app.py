import gradio as gr
import torch
from PIL import Image
import os

from model.fetalnet import FetalNet
from utils.preprocess import transform_image
from utils.prediction import make_prediction

# Load the Model once
def load_model():
    device = torch.device("cpu")
    model = FetalNet(num_classes_model=6).to(device)
    model_path = "model/trained_models/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()
print("Model loaded successfully.")

# Main function for Gradio
def predict(input_image):
    tensor = transform_image(input_image)
    _, _, all_probs = make_prediction(model, tensor)
    return all_probs

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fetal Ultrasound Classifier") as demo:
    gr.Markdown("Fetal Ultrasound Plane Classifier 🔬")
    gr.Markdown("Upload a fetal ultrasound image to see the model's classification confidence for each anatomical plane.")
    
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Ultrasound Image")
        prediction_label = gr.Label(label="Prediction Confidence")

    submit_btn = gr.Button("Classify Plane", variant="primary")
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=prediction_label,
        api_name="predict"
    )
    
    gr.Examples(
        examples=[os.path.join("examples", "example_brain.png"), os.path.join("examples", "example_abdomen.png")],
        inputs=input_image,
    )

if __name__ == "__main__":
    demo.launch()