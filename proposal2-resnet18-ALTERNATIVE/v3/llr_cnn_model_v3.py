"""
llr_cnn_model_v3.py

Version 3 model: UNet architecture.
Predicts 6 heatmaps (one per landmark) instead of (x, y) coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LLRLandmarkCNNv3(nn.Module):
    def __init__(self):
        super(LLRLandmarkCNNv3, self).__init__()

        # --- Encoder ---
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        # --- Decoder ---
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(384, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(192, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(96, 32)

        # --- Output: 6 heatmaps ---
        self.final_conv = nn.Conv2d(32, 6, kernel_size=1)  # 6 heatmaps output

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Decoder
        u4 = self.upconv4(p4)
        u4 = torch.cat((u4, e4), dim=1)
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(u2)

        # Final output: 6 heatmaps
        out = self.final_conv(d2)
        return out
