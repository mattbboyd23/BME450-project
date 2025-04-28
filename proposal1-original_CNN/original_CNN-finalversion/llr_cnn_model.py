# File: llr_cnn_model.py
"""
Original CNN vFinal:
- Four 5x5 conv layers, pooling, FC head
- Sigmoid output for normalized coords
"""
import torch.nn as nn
import torch.nn.functional as F

class LLRLandmarkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64,128, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(2, 2)
        # After 4 pools: 640×192 → 40×12
        self.fc1   = nn.Linear(128 * 40 * 12, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 12)
        self.act   = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.act(self.fc3(x))