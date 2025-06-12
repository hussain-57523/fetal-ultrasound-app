# utils/gradcam_utils.py
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, input_tensor, target_layer, target_category=None, device='cpu'):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == 'cuda'))

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])
    grayscale_cam = grayscale_cam[0, :]
    input_image_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    input_image_np = (input_image_np * 0.5 + 0.5)  # De-normalize
    input_image_np = np.clip(input_image_np, 0, 1)

    cam_image = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)
    return cam_image
