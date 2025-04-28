"""
llr_cnn_model_v2_1_light.py

Lightweight UNet model for faster training/testing.
Smaller feature maps, fewer channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LLRLandmarkCNNv2_1_Light(nn.Module):
    def __init__(self):
        super(LLRLandmarkCNNv2_1_Light, self).__init__()

        # --- Encoder ---
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)

        self.pool = nn.MaxPool2d(2)

        # --- Decoder ---
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(192, 64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(96, 32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(48, 16)

        # --- Final output layers ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 224 * 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        u4 = self.upconv4(p4)
        u4 = torch.cat((u4, e4), dim=1)
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(u2)

        x = self.flatten(d2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # normalized output
        return x
