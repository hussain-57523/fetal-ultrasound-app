import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Squeeze-and-Excitation Block as used in the successful model
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# The correct FetalNet architecture from your training notebook
class FetalNet(nn.Module):
    def __init__(self, num_classes_model=6): # Corrected to 6 classes
        super(FetalNet, self).__init__()
        
        # Use the modern 'weights' argument and freeze the base model
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        for param in self.base.parameters():
            param.requires_grad = False
            
        # This is the correct custom head with two Conv stages and the SE block
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(256), # SE Block is here
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # This is the correct classifier structure
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes_model)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x