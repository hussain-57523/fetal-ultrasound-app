import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class FetalNet(nn.Module):
    def __init__(self, num_classes_model=6):
        super(FetalNet, self).__init__()
        self.base = models.mobilenet_v2(weights=None).features
        for param in self.base.parameters():
            param.requires_grad = False
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            SEBlock(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, num_classes_model)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x