# File: llr_cnn_model_v3_3.py
"""
CNN model v3_3:
- Pretrained ResNet18 backbone adapted to single-channel input
- Light regression head with dropout and Sigmoid
"""
import torch
import torch.nn as nn
import torchvision.models as models

class LLRLandmarkCNN_v3_3(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained resnet18 and adapt conv1 for 1-channel
        backbone = models.resnet18(pretrained=True)
        # replace conv1: in_channels=1, preserve pretrained weights by averaging
        orig_w = backbone.conv1.weight.data
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = orig_w.mean(dim=1, keepdim=True)

        # Remove original fc, add custom regression head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 12),
            nn.Sigmoid(),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)
