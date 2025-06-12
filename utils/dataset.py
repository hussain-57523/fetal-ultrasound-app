import pandas as pd # If you use pandas for CSV reading
import os # If you use os.path functions

# fetal_ultrasound_project/utils/dataset.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class FetalDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {
            'Fetal abdomen': 0, 'Fetal brain': 1, 'Fetal femur': 2,
            'Fetal thorax': 3, 'Maternal cervix': 4, 'Other': 5
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['image_name'])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.dataframe.iloc[idx]['plane_type']]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Standard Transformations (can be imported or defined here) ---
# Ensure these match the transformations used during your training
def get_train_transforms(image_size, norm_mean, norm_std):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

def get_val_test_transforms(image_size, norm_mean, norm_std):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

def get_inference_transforms(image_size, norm_mean, norm_std):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])