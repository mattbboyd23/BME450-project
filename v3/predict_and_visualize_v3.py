"""
predict_and_visualize_v3.py

Loads trained v3 UNet model, predicts heatmaps,
and extracts (x, y) landmark coordinates by finding peaks.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from llr_cnn_model_v3 import LLRLandmarkCNNv3
from llr_dataset import LLRDataset
from heatmap_utils import generate_target_heatmaps  # Only if needed
import os

# === PATH SETUP ===
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
model_path = os.path.join(base_path, 'v3', 'llr_model_v3.pth')
excel_file = os.path.join(base_path, 'v3', 'outputs.xlsx')
image_dir = os.path.join(base_path, 'v3', 'data_acquisition', 'raw_data')

# === SETTINGS ===
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 192
HEATMAP_HEIGHT = 80
HEATMAP_WIDTH = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = LLRLandmarkCNNv3().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === LOAD ONE SAMPLE ===
transform = torch.nn.Sequential(
    # No augmentation for testing
    torch.nn.Upsample(size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False)
)

dataset = LLRDataset(excel_file=excel_file, image_dir=image_dir, transform=None)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# === PREDICT ===
with torch.no_grad():
    for images, coords in loader:
        images = images.to(device)

        outputs = model(images)

        # Resize outputs to match heatmap size
        outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)

        # outputs shape: [1, 6, H, W]
        heatmaps = outputs.squeeze(0).cpu()

        predicted_coords = []

        for heatmap in heatmaps:
            max_val = heatmap.max()
            y, x = (heatmap == max_val).nonzero(as_tuple=True)
            x = x.item()
            y = y.item()

            # Rescale back to original image size
            x_scaled = (x / HEATMAP_WIDTH) * IMAGE_WIDTH
            y_scaled = (y / HEATMAP_HEIGHT) * IMAGE_HEIGHT

            predicted_coords.append((x_scaled, y_scaled))

        print("Predicted Landmark Coordinates (pixels):")
        for i, (x, y) in enumerate(predicted_coords):
            print(f"Landmark {i+1}: (x={x:.1f}, y={y:.1f})")

        # OPTIONAL: Visualize one predicted heatmap
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()
        for i, heatmap in enumerate(heatmaps):
            axs[i].imshow(heatmap.numpy(), cmap='hot')
            axs[i].set_title(f"Landmark {i+1}")
            axs[i].axis('off')
        plt.show()

        break  # Predict one sample for now
