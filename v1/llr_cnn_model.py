"""
llr_cnn_model.py

This file defines a convolutional neural network (CNN) model to predict
six anatomical landmark locations from long-leg radiograph (LLR) images.

Input:
    - Grayscale image tensor of shape [1, 896, 256]

Output:
    - Tensor of shape [12] containing predicted (x, y) coordinates for:
      RH, RK, RA, LH, LK, LA (in that order)

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class LLRLandmarkCNN(nn.Module):
    def __init__(self):
        super(LLRLandmarkCNN, self).__init__()

        # Use 5x5 kernels with padding=2 to preserve spatial size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 56 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 12)  # 6 landmarks × (x, y)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → [16, 448, 128]
        x = self.pool(F.relu(self.conv2(x)))  # → [32, 224, 64]
        x = self.pool(F.relu(self.conv3(x)))  # → [64, 112, 32]
        x = self.pool(F.relu(self.conv4(x)))  # → [128, 56, 16]

        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
