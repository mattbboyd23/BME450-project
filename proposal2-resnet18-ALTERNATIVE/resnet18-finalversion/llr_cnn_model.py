# File: llr_cnn_model.py
"""
ResNet18-based landmark detector:
- Pretrained 1-channel ResNet18 backbone, freeze low-level features
- Small FC head, 6x(x,y) outputs in [0,1]
"""
import torch.nn as nn
import torchvision.models as models

class LLRLandmarkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        # adapt first conv to 1-channel
        w0 = backbone.conv1.weight.data
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = w0.mean(dim=1, keepdim=True)
        # freeze low-level features
        for param in backbone.layer1.parameters():
            param.requires_grad = False
        # replace head
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
