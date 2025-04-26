"""
overlay_landmarks_v3.py

Loads an image, predicts landmarks using v3 model,
and overlays the landmarks as yellow dots.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from llr_cnn_model_v3 import LLRLandmarkCNNv3
from llr_dataset import LLRDataset
import os

# === PATH SETUP ===
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
model_path = os.path.join(base_path, 'v3', 'llr_model_v3.pth')
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

# === LOAD ONE IMAGE MANUALLY ===
image_name = 'sample23-9565513-resized.jpg'  # <-- Change this to your image filename!
image_path = os.path.join(image_dir, image_name)

img = Image.open(image_path).convert('L')
img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
img_tensor = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img_resized.tobytes())), dtype=torch.uint8)
img_tensor = img_tensor.view(IMAGE_HEIGHT, IMAGE_WIDTH).unsqueeze(0).unsqueeze(0).float() / 255.0

img_tensor = img_tensor.to(device)

# === PREDICT ===
with torch.no_grad():
    outputs = model(img_tensor)

    outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)
    heatmaps = outputs.squeeze(0).cpu()

    predicted_coords = []

    for heatmap in heatmaps:
        max_val = heatmap.max()
        y, x = (heatmap == max_val).nonzero(as_tuple=True)
        x = x.item()
        y = y.item()

        # Rescale back to image coordinates
        x_scaled = (x / HEATMAP_WIDTH) * IMAGE_WIDTH
        y_scaled = (y / HEATMAP_HEIGHT) * IMAGE_HEIGHT

        predicted_coords.append((x_scaled, y_scaled))

# === PLOT IMAGE + LANDMARKS ===
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(img_resized, cmap='gray')
for (x, y) in predicted_coords:
    ax.plot(x, y, 'yo', markersize=6)  # yellow dots
ax.set_title('Predicted Landmarks Overlayed')
ax.axis('off')
save_path = os.path.join(base_path, 'v3', 'predicted_overlay.png')
plt.savefig(save_path)
print(f"Saved overlay to {save_path}")