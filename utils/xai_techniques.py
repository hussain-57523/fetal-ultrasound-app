import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients, Occlusion

# This list is used for plot titles and must match the model's training order
CLASS_NAMES = [
    'Fetal abdomen', 'Fetal brain', 'Fetal femur', 
    'Fetal thorax', 'Maternal cervix', 'Other'
]

def get_prediction_index(model, input_tensor):
    """Helper function to get the model's top prediction index."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return torch.argmax(output).item()

def generate_grad_cam(model, input_tensor, original_image_pil):
    """Generates the Grad-CAM visualization."""
    try:
        target_layer = [model.conv[5]]
        cam = GradCAM(model=model, target_layers=target_layer)
        pred_index = get_prediction_index(model, input_tensor)
        targets = [ClassifierOutputTarget(pred_index)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_img_float = np.array(original_image_pil.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        return visualization, CLASS_NAMES[pred_index]
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        return None, "Error"

def generate_guided_backprop(model, device, input_tensor):
    """Generates the Guided Backpropagation visualization."""
    try:
        pred_index = get_prediction_index(model, input_tensor)
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
        gradients = gb_model(input_tensor, target_category=pred_index)
        gb_viz = np.transpose(gradients.squeeze(), (1, 2, 0))
        gb_viz = (gb_viz - gb_viz.min()) / (gb_viz.max() - gb_viz.min() + 1e-9)
        return gb_viz, CLASS_NAMES[pred_index]
    except Exception as e:
        print(f"Error in Guided Backprop: {e}")
        return None, "Error"