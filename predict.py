# predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.custom_cnn import FetalNet

def load_model(weights_path, device, num_classes=6):
    model = FetalNet(num_classes_model=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_file).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension
