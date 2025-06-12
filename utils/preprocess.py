# utils/preprocess.py

from torchvision import transforms
from PIL import Image

# Define the image transformations. This should match your validation/test transform.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def transform_image(image_bytes):
    """
    Takes the bytes of an uploaded image, converts it to a PIL Image,
    applies the necessary transformations, and returns a tensor
    ready for the model.
    """
    try:
        # Open the image, convert to grayscale then to RGB to match training
        image = Image.open(image_bytes).convert("L").convert("RGB")
        # Apply the transformations and add a batch dimension
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error transforming image: {e}")
        return None