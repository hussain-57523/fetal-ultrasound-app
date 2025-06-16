import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import (
    IntegratedGradients, 
    Occlusion, 
    GuidedBackprop, 
    LayerGradCam,
    visualization as viz
)

from utils.prediction import CLASS_NAMES

def get_prediction_index(model, input_tensor):
    """Helper to get the model's top prediction index."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return torch.argmax(output).item()

def generate_gradcam_figure(model, input_tensor, original_image):
    """Generates LayerGradCam visualization as a matplotlib figure."""
    target_layer = model.conv[5]
    gradcam = LayerGradCam(model, target_layer)
    pred_index = get_prediction_index(model, input_tensor)
    
    attributions = gradcam.attribute(input_tensor, target=pred_index)
    upsampled_attr = torch.nn.functional.interpolate(attributions, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Use Captum's visualization tool which returns a figure
    fig, _ = viz.visualize_image_attr(
        np.transpose(upsampled_attr.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.array(original_image.resize((224, 224))),
        method='blended_heat_map', sign='positive', cmap='viridis',
        show_colorbar=False, title="Grad-CAM: Model Focus"
    )
    return fig

# You can add the other XAI functions here as needed (GuidedBackprop, etc.) following this pattern.
# For simplicity in the main app, we will focus on just Grad-CAM for now.