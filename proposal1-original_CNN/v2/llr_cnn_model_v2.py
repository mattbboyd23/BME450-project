"""
llr_cnn_model_v2.py

Version 2 of the CNN model using a UNet-style encoder with skip-like features.
Predicts 12 normalized landmark coordinates (values between 0 and 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LLRLandmarkCNNv2(nn.Module):
    def __init__(self):
        super(LLRLandmarkCNNv2, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 896x256 → 448x128
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # → 224x64
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # → 112x32
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # → 56x16
        )

        # Bottleneck → Fully connected
        self.flattened_size = 256 * 56 * 16
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 12)  # 6 landmarks × (x, y)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Normalized output (0 to 1)
        return x
