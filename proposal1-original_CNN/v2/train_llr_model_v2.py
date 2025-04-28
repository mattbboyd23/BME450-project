"""
train_llr_model_v2.py

Trains the v2 UNet-style CNN using normalized landmark outputs.
Includes contrast enhancement, grouped MSE reporting, and LR scheduler.
"""

from llr_dataset import LLRDataset
from llr_cnn_model_v2 import LLRLandmarkCNNv2

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
excel_file = os.path.join(base_path, 'v2', 'outputs.xlsx')
image_dir = os.path.join(base_path, 'v2', 'data_acquisition', 'raw_data')
model_save_path = os.path.join(base_path, 'v2', 'llr_model_v2.pth')

# === TRAINING SETTINGS ===
IMAGE_HEIGHT = 896
IMAGE_WIDTH = 256
NUM_EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(2.0)),  # contrast boost
    transforms.ToTensor()
])

# === DATASET ===
dataset = LLRDataset(excel_file=excel_file, image_dir=image_dir, transform=transform)

# Normalize ground truth coordinates to [0, 1] during training
def normalize_coords(coords):
    coords = coords.clone()
    coords[::2] /= IMAGE_WIDTH   # x
    coords[1::2] /= IMAGE_HEIGHT # y
    return coords

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === MODEL + LOSS + OPTIMIZER ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LLRLandmarkCNNv2().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# === TRAINING LOOP ===
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = normalize_coords(targets).to(device)  # normalize to 0–1

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    hip_loss = 0.0
    knee_loss = 0.0
    ankle_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = normalize_coords(targets).to(device)
            outputs = model(images)

            val_loss += criterion(outputs, targets).item()

            # Rescale to pixel space for group loss analysis
            pred_px = outputs * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT]*6, device=device)
            true_px = targets * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT]*6, device=device)

            for i in range(images.size(0)):
                pred = pred_px[i]
                true = true_px[i]

                hip_pred = torch.cat([pred[0:2], pred[6:8]])
                hip_true = torch.cat([true[0:2], true[6:8]])
                hip_loss += F.mse_loss(hip_pred, hip_true).item()

                knee_pred = torch.cat([pred[2:4], pred[8:10]])
                knee_true = torch.cat([true[2:4], true[8:10]])
                knee_loss += F.mse_loss(knee_pred, knee_true).item()

                ankle_pred = torch.cat([pred[4:6], pred[10:12]])
                ankle_true = torch.cat([true[4:6], true[10:12]])
                ankle_loss += F.mse_loss(ankle_pred, ankle_true).item()

    scheduler.step()

    avg_val_loss = val_loss / len(val_loader)
    avg_hip_loss = hip_loss / len(val_dataset)
    avg_knee_loss = knee_loss / len(val_dataset)
    avg_ankle_loss = ankle_loss / len(val_dataset)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f}")
    print(f"    Hip MSE: {avg_hip_loss:.2f} | Knee MSE: {avg_knee_loss:.2f} | Ankle MSE: {avg_ankle_loss:.2f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")
