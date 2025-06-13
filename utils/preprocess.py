# utils/preprocess.py

from torchvision import transforms
from PIL import Image

# Image transformations - must match what the model was trained with
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def transform_image(image_bytes):
    """
    Transforms image bytes into a model-ready tensor.
    """
    try:
        # Open the image, ensure it's in RGB format
        image = Image.open(image_bytes).convert("RGB")
        # Apply the transformations and add a batch dimension
        return image_transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error transforming image: {e}")
        return None