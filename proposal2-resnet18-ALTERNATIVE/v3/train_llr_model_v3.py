"""
train_llr_model_v3.py

Training script for v3 UNet model that predicts 6 heatmaps (one per landmark).
Uses Gaussian heatmaps as training targets.
"""

from llr_dataset import LLRDataset
from llr_cnn_model_v3 import LLRLandmarkCNNv3
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
excel_file = os.path.join(base_path, 'v3', 'outputs.xlsx')
image_dir = os.path.join(base_path, 'v3', 'data_acquisition', 'raw_data')
model_save_path = os.path.join(base_path, 'v3', 'llr_model_v3.pth')

# === TRAINING SETTINGS ===
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 192
HEATMAP_HEIGHT = 80    # Downscaled version (640/8)
HEATMAP_WIDTH = 24     # Downscaled version (192/8)
NUM_EPOCHS = 5 # Number of epochs to train
BATCH_SIZE = 1
LEARNING_RATE = 5e-5

# === IMAGE TRANSFORM WITH AUGMENTATION ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(contrast=0.2),
    transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(2.0)),
    transforms.ToTensor()
])

# === DATASET ===
dataset = LLRDataset(excel_file=excel_file, image_dir=image_dir, transform=transform)

def normalize_coords(coords):
    coords = coords.clone()
    coords[::2] *= (192 / 256)
    coords[1::2] *= (640 / 896)
    coords[::2] /= 192
    coords[1::2] /= 640
    return coords

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === MODEL, LOSS, OPTIMIZER ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LLRLandmarkCNNv3().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# === TRAINING LOOP ===
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for images, coords in train_loader:
        images = images.to(device)
        coords = normalize_coords(coords).to(device)
        coords = torch.clamp(coords, 0.0, 1.0)


        # Generate target heatmaps
        target_heatmaps = []
        for i in range(images.size(0)):
            heatmap = generate_target_heatmaps(coords[i], (HEATMAP_HEIGHT, HEATMAP_WIDTH))
            target_heatmaps.append(heatmap)

        target_heatmaps = torch.stack(target_heatmaps).to(device)

        # Forward pass
        outputs = model(images)

        # Resize outputs to match heatmap size
        outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)

        loss = criterion(outputs, target_heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, coords in val_loader:
            images = images.to(device)
            coords = normalize_coords(coords).to(device)
            coords = torch.clamp(coords, 0.0, 1.0)

            target_heatmaps = []
            for i in range(images.size(0)):
                heatmap = generate_target_heatmaps(coords[i], (HEATMAP_HEIGHT, HEATMAP_WIDTH))
                target_heatmaps.append(heatmap)

            target_heatmaps = torch.stack(target_heatmaps).to(device)

            outputs = model(images)
            outputs = F.interpolate(outputs, size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), mode='bilinear', align_corners=False)

            loss = criterion(outputs, target_heatmaps)
            val_loss += loss.item()

    scheduler.step()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")
