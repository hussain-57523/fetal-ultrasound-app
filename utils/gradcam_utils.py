# utils/gradcam_utils.py

import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayer

# This list is used for plot titles
CLASS_NAMES = [
    'Fetal abdomen', 'Fetal brain', 'Fetal femur', 
    'Fetal thorax', 'Maternal cervix', 'Other'
]

def get_prediction_details(model, input_tensor):
    """Helper function to get the model's top prediction index."""
    output = model(input_tensor)
    return torch.argmax(output).item()

def generate_grad_cam(model, input_tensor, original_image_pil):
    """Generates the Grad-CAM visualization."""
    target_layer = [model.conv[5]] # Last Conv2D layer
    cam = GradCAM(model=model, target_layers=target_layer)
    
    pred_index = get_prediction_details(model, input_tensor)
    targets = [ClassifierOutputTarget(pred_index)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    rgb_img_float = np.array(original_image_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    return visualization, CLASS_NAMES[pred_index]

def generate_guided_backprop(model, input_tensor):
    """Generates the Guided Backpropagation visualization."""
    pred_index = get_prediction_details(model, input_tensor)
    
    gb_model = GuidedBackpropReLUModel(model=model, device=torch.device('cpu')) # Forcing CPU for this part can sometimes be more stable
    gradients = gb_model(input_tensor, target_category=pred_index)
    
    # Process for visualization
    gb_viz = gradients.squeeze()
    if gb_viz.shape[0] == 3: # Handle channel-first format
        gb_viz = gb_viz.transpose(1, 2, 0)
        
    gb_viz = (gb_viz - gb_viz.min()) / (gb_viz.max() - gb_viz.min() + 1e-9)
    return gb_viz, CLASS_NAMES[pred_index]

def generate_occlusion_sensitivity(model, input_tensor, original_image_pil):
    """Generates the Occlusion Sensitivity visualization. This is slow."""
    ablation_layer = AblationLayer(model)
    cam = AblationCAM(model=model, target_layers=[ablation_layer], ablation_layer=ablation_layer)

    pred_index = get_prediction_details(model, input_tensor)
    targets = [ClassifierOutputTarget(pred_index)]
    
    occlusion_map = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # For visualization, we can just return the heatmap itself
    # Matplotlib can overlay it. We'll do the overlay in the app itself.
    return occlusion_map, CLASS_NAMES[pred_index]

def generate_integrated_gradients(model, input_tensor):
    """Generates the Integrated Gradients visualization."""
    from pytorch_grad_cam.fullgrad import FullGrad # Import locally to avoid potential conflicts
    
    integrated_gradients = FullGrad(model)
    
    pred_index = get_prediction_details(model, input_tensor)
    targets = [ClassifierOutputTarget(pred_index)]
    
    attribution_map = integrated_gradients.attribute(input_tensor, targets, n_steps=50)
    
    # Process for visualization
    ig_viz = attribution_map.squeeze()
    if ig_viz.shape[0] == 3: # Handle channel-first format
        ig_viz = ig_viz.transpose(1, 2, 0)
        
    ig_viz = (ig_viz - ig_viz.min()) / (ig_viz.max() - ig_viz.min() + 1e-9)
    return ig_viz.cpu().detach().numpy(), CLASS_NAMES[pred_index]