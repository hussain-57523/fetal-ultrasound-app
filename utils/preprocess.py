# utils/preprocess.py

import torchvision.transforms as transforms
from PIL import Image
import torch
from utils.preprocess import preprocess_image

# Define the transformation used for the test image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses the input PIL image and returns a tensor ready for model input.
    """
    return transform(image).unsqueeze(0)  # Add batch dimension
