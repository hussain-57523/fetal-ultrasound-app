import torchvision.transforms as transforms
from PIL import Image
import torch

# This is the correct transformation pipeline from your training notebook
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# The function is now correctly named 'transform_image'
def transform_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses the input PIL image and returns a tensor ready for model input.
    """
    # The self-import line has been removed.
    image_rgb = image.convert("RGB")
    return image_transform(image_rgb).unsqueeze(0)  # Add batch dimension