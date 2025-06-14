from torchvision import transforms
from PIL import Image

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def transform_image(pil_image):
    try:
        image = pil_image.convert("RGB")
        return image_transform(image).unsqueeze(0)
    except Exception as e:
        return None