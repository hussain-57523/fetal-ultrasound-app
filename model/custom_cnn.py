from model.se_block import SEBlock # Import SEBlock from its own file# fetal_ultrasound_project/model/custom_cnn.py

import torch
import torch.nn as nn
import torchvision
from model.se_block import SEBlock # Import SEBlock

class FetalNet(nn.Module):
    def __init__(self, num_classes=6):
        super(FetalNet, self).__init__()
        
        # Load pre-trained MobileNetV2 (weights are only used for architecture definition here,
        # but specifying them for consistency with training)
        self.base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        
        # Freeze base model parameters (important for consistency during inference)
        for param in self.base.parameters():
            param.requires_grad = False
            
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            SEBlock(256), # Your custom SEBlock

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout is inactive in model.eval()
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x