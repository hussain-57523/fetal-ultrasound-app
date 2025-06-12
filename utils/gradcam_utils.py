import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam_visualizations(model, dataset, target_layer, target_names, device, num_images=5):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))

    for i in range(num_images):
        image_tensor, label_idx = dataset[i]
        unnormalized_img = image_tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        unnormalized_img = std * unnormalized_img + mean
        unnormalized_img = np.clip(unnormalized_img, 0, 1)

        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx_tensor = torch.max(output, 1)
            predicted_idx = predicted_idx_tensor.item()

        targets = [ClassifierOutputTarget(predicted_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        visualization = show_cam_on_image(unnormalized_img, grayscale_cam, use_rgb=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Image #{i} | True: {target_names[label_idx]} | Pred: {target_names[predicted_idx]}', fontsize=14)
        ax1.imshow(unnormalized_img)
        ax1.set_title('Original')
        ax1.axis('off')

        ax2.imshow(visualization)
        ax2.set_title('Grad-CAM')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()
