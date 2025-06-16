from torchvision import transforms
from PIL import Image
import torch

# This transformation pipeline MUST match what was used during training
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses an input PIL image and returns a tensor ready for the model.
    """
    image_rgb = image.convert("RGB")
    return image_transform(image_rgb).unsqueeze(0) # Add batch dimension