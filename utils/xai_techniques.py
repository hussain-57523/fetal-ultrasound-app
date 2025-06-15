import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from captum.attr import (
    IntegratedGradients, 
    Occlusion, 
    GuidedBackprop, 
    LayerGradCam,
    visualization as viz
)

# --- 1. Correct Class Names (Must match your model's training) ---
CLASS_NAMES = [
    'Fetal abdomen', 'Fetal brain', 'Fetal femur', 
    'Fetal thorax', 'Maternal cervix', 'Other'
]

def get_prediction(model, input_tensor):
    """Helper function to get the model's top prediction index and name."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_index = torch.argmax(output).item()
        pred_name = CLASS_NAMES[pred_index]
    return pred_index, pred_name

def process_attributions(attributions):
    """Helper to process attribution maps for visualization."""
    attr_np = attributions.squeeze().cpu().detach().numpy()
    # Handle both grayscale and RGB attributions
    if attr_np.shape[0] == 3: # If shape is (C, H, W)
        attr_np = np.transpose(attr_np, (1, 2, 0))
    
    # Normalize the values to be in the 0-1 range for display
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-9)
    return attr_np

# --- 2. Updated XAI Functions ---
# Each function now takes the model as an argument and explains the actual prediction.

def generate_gradcam(model, input_tensor, original_image):
    """Generates LayerGradCam visualization."""
    # Correctly target the last convolutional layer in your custom FetalNet
    target_layer = model.conv[5] 
    gradcam = LayerGradCam(model, target_layer)
    
    pred_index, _ = get_prediction(model, input_tensor)
    
    attributions = gradcam.attribute(input_tensor, target=pred_index)
    # Upsample the heatmap to the image size
    upsampled_attr = torch.nn.functional.interpolate(
        attributions, size=(224, 224), mode='bilinear', align_corners=False
    )
    heatmap = upsampled_attr.squeeze().cpu().detach().numpy()
    
    # Use Captum's visualization tool to create the overlay
    fig, _ = viz.visualize_image_attr(
        np.transpose(heatmap, (1, 2, 0)), # Must be (H, W, C)
        np.array(original_image.resize((224, 224))),
        method='blended_heat_map',
        sign='positive',
        show_colorbar=True,
        title="Grad-CAM Overlay"
    )
    return fig

def generate_guided_backprop(model, input_tensor):
    """Generates Guided Backpropagation visualization."""
    gbp = GuidedBackprop(model)
    pred_index, _ = get_prediction(model, input_tensor)
    attributions = gbp.attribute(input_tensor, target=pred_index)
    
    # Process for display
    processed_attr = process_attributions(attributions)
    return processed_attr

def generate_integrated_gradients(model, input_tensor):
    """Generates Integrated Gradients visualization."""
    ig = IntegratedGradients(model)
    pred_index, _ = get_prediction(model, input_tensor)
    
    # Use a black image as a baseline
    baseline = torch.zeros_like(input_tensor)
    attributions = ig.attribute(input_tensor, baselines=baseline, target=pred_index, n_steps=50)
    
    # Process for display
    processed_attr = process_attributions(attributions)
    # For IG, often we visualize the max absolute value across channels
    return np.max(np.abs(processed_attr), axis=2)

def generate_occlusion(model, input_tensor):
    """Generates Occlusion visualization. NOTE: This is slow."""
    occlusion = Occlusion(model)
    pred_index, _ = get_prediction(model, input_tensor)
    
    attributions = occlusion.attribute(input_tensor,
                                     strides=(3, 8, 8),
                                     target=pred_index,
                                     sliding_window_shapes=(3, 15, 15),
                                     baselines=0)
    
    # Process for display
    processed_attr = process_attributions(attributions)
    # For Occlusion, we visualize the max absolute value across channels
    return np.max(np.abs(processed_attr), axis=2)