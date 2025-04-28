# File: llr_cnn_model_v3_4.py
"""
Model v3_4:
- Pretrained ResNet18 backbone adapted to 1-channel
- Freeze layer1 to focus feature tuning
- Regression head with dropout + Sigmoid
"""
import torch
import torch.nn as nn
import torchvision.models as models

class LLRLandmarkCNN_v3_4(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        # adapt conv1 to single-channel
        w0 = backbone.conv1.weight.data
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = w0.mean(dim=1, keepdim=True)
        # freeze first residual block
        for param in backbone.layer1.parameters(): param.requires_grad = False
        # replace fc
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 12),
            nn.Sigmoid(),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)
