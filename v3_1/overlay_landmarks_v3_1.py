"""
overlay_landmarks_v3_1.py

Final clean overlay script:
- Predicts from trained model
- Extracts landmark (x, y) locations from heatmaps
- Correctly rescales for resized 192x640 images
- Saves overlay plot
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from llr_cnn_model_v3_1 import LLRLandmarkCNNv3_1
import os

# === PATH SETUP ===
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
model_path = os.path.join(base_path, 'v3_1', 'llr_model_v3_1.pth')
image_dir = os.path.join(base_path, 'v3_1', 'data_acquisition', 'raw_data')

# === SETTINGS ===
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 192
HEATMAP_HEIGHT = 80
HEATMAP_WIDTH = 24
landmark_names = ['RH', 'RK', 'RA', 'LH', 'LK', 'LA']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = LLRLandmarkCNNv3_1().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === LOAD IMAGE ===
image_name = 'sample23-9565513-resized.jpg'  # <-- change this to test different images
image_path = os.path.join(image_dir, image_name)

img = Image.open(image_path).convert('L')
img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
img_tensor = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())),
                          dtype=torch.uint8)
img_tensor = img_tensor.view(IMAGE_HEIGHT, IMAGE_WIDTH).unsqueeze(0).unsqueeze(0).float() / 255.0
img_tensor = img_tensor.to(device)

# === PREDICT ===
with torch.no_grad():
    outputs = model(img_tensor)
    outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)
    heatmaps = outputs.squeeze(0).cpu()  # shape: [6, H, W]

    predicted_coords = []

    for i, heatmap in enumerate(heatmaps):
        max_val = heatmap.max().item()
        if max_val < 1e-6:
            x_scaled, y_scaled = -1, -1
        else:
            y, x = (heatmap == max_val).nonzero(as_tuple=True)
            if len(x) == 0 or len(y) == 0:
                x_scaled, y_scaled = -1, -1
            else:
                x = x[0].item()
                y = y[0].item()
                # Correct scaling math
                x_scaled = (x / (HEATMAP_WIDTH - 1)) * (IMAGE_WIDTH - 1)
                y_scaled = (y / (HEATMAP_HEIGHT - 1)) * (IMAGE_HEIGHT - 1)

        predicted_coords.append((x_scaled, y_scaled))

# === PLOT OVERLAY ===
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(img, cmap='gray')

for i, (x, y) in enumerate(predicted_coords):
    if x >= 0 and y >= 0:
        ax.plot(x, y, 'yo', markersize=6)
        ax.text(x + 3, y, landmark_names[i], color='yellow', fontsize=8, fontweight='bold')

ax.set_title('Predicted Landmarks Overlayed (v3_1)')
ax.axis('off')

# === SAVE OVERLAY ===
save_path = os.path.join(base_path, 'v3_1', 'predicted_overlay_v3_1.png')
plt.savefig(save_path)
print(f"Overlay saved to: {save_path}")
