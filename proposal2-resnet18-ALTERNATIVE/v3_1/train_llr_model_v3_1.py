"""
train_llr_model_v3_1.py

 training script for v3_1:
- Loads from correct grouped dataset
- Normalizes coordinates during training
- Trains CNN to predict landmark heatmaps
"""

from llr_dataset_v3_1 import LLRDataset
from llr_cnn_model_v3_1 import LLRLandmarkCNNv3_1
from heatmap_utils import generate_target_heatmaps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import ImageEnhance
import os

# === PATH SETUP ===
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
excel_file = os.path.join(base_path, 'v3_1', 'outputs.xlsx')
image_dir = os.path.join(base_path, 'v3_1', 'data_acquisition', 'raw_data')
model_save_path = os.path.join(base_path, 'v3_1', 'llr_model_v3_1.pth')

# === SETTINGS ===
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 192
HEATMAP_HEIGHT = 80
HEATMAP_WIDTH = 24
NUM_EPOCHS = 25
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
landmark_names = ['RH', 'RK', 'RA', 'LH', 'LK', 'LA']

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(2.0)),
    transforms.ToTensor()
])

# === DATASET ===
dataset = LLRDataset(excel_file=excel_file, image_dir=image_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === MODEL, LOSS, OPTIMIZER ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LLRLandmarkCNNv3_1().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# === TRAINING LOOP ===
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    landmark_losses = [0.0 for _ in range(6)]

    for images, coords in train_loader:
        images = images.to(device)
        coords = coords.to(device)

        # Normalize coords between 0–1 relative to resized image
        coords_norm = coords.clone()
        coords_norm[::2] /= IMAGE_WIDTH  # x coords
        coords_norm[1::2] /= IMAGE_HEIGHT  # y coords
        coords_norm = torch.clamp(coords_norm, 0.0, 1.0)


        # Generate heatmaps
        target_heatmaps = []
        for i in range(images.size(0)):
            heatmap = generate_target_heatmaps(coords_norm[i], (HEATMAP_HEIGHT, HEATMAP_WIDTH))
            target_heatmaps.append(heatmap)

        target_heatmaps = torch.stack(target_heatmaps).to(device)

        # Forward
        outputs = model(images)
        outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)

        loss = 0.0
        per_landmark_losses = []

        for i in range(6):
            l = criterion(outputs[:, i, :, :], target_heatmaps[:, i, :, :])
            loss += l
            per_landmark_losses.append(l.item())

        loss = loss / 6

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        for i in range(6):
            landmark_losses[i] += per_landmark_losses[i]

    avg_train_loss = train_loss / len(train_loader)
    avg_landmark_losses = [l / len(train_loader) for l in landmark_losses]

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, coords in val_loader:
            images = images.to(device)
            coords = coords.to(device)

            coords_norm = coords.clone()
            coords_norm[::2] /= IMAGE_WIDTH
            coords_norm[1::2] /= IMAGE_HEIGHT
            coords_norm = torch.clamp(coords_norm, 0.0, 1.0)


            target_heatmaps = []
            for i in range(images.size(0)):
                heatmap = generate_target_heatmaps(coords_norm[i], (HEATMAP_HEIGHT, HEATMAP_WIDTH))
                target_heatmaps.append(heatmap)

            target_heatmaps = torch.stack(target_heatmaps).to(device)

            outputs = model(images)
            outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)

            loss = criterion(outputs, target_heatmaps)
            val_loss += loss.item()

    scheduler.step()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    for i, landmark_loss in enumerate(avg_landmark_losses):
        print(f"    {landmark_names[i]} Train Loss: {landmark_loss:.6f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")
