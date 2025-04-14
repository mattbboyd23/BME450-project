"""
train_llr_model.py

This script trains a CNN to predict 6 anatomical landmarks (12 coordinates)
from grayscale long-leg radiograph (LLR) images.

Components:
    - Loads images and coordinates using LLRDataset
    - Loads model from llr_cnn_model.py
    - Trains using MSE loss (mean squared error)
    - Saves model checkpoint after training
"""
import torch.nn.functional as F
from llr_dataset import LLRDataset
from llr_cnn_model import LLRLandmarkCNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

# === PATHS & DATASET === #
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT'
excel_file = os.path.join(base_path, 'outputs.xlsx')
image_dir = base_path

transform = transforms.Compose([
    transforms.Resize((896, 256)),
    transforms.ToTensor()
])

# Load dataset and split into train/val
full_dataset = LLRDataset(excel_file=excel_file, image_dir=image_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# === MODEL SETUP === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LLRLandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === TRAINING LOOP === #
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # === VALIDATION WITH GROUPED MSEs === #
    model.eval()
    val_loss = 0.0
    hip_loss = 0.0
    knee_loss = 0.0
    ankle_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, targets).item()

            for i in range(images.size(0)):
                pred = outputs[i]
                true = targets[i]

                # Hip: RH (0:2), LH (6:8)
                hip_pred = torch.cat([pred[0:2], pred[6:8]])
                hip_true = torch.cat([true[0:2], true[6:8]])
                hip_loss += F.mse_loss(hip_pred, hip_true).item()

                # Knee: RK (2:4), LK (8:10)
                knee_pred = torch.cat([pred[2:4], pred[8:10]])
                knee_true = torch.cat([true[2:4], true[8:10]])
                knee_loss += F.mse_loss(knee_pred, knee_true).item()

                # Ankle: RA (4:6), LA (10:12)
                ankle_pred = torch.cat([pred[4:6], pred[10:12]])
                ankle_true = torch.cat([true[4:6], true[10:12]])
                ankle_loss += F.mse_loss(ankle_pred, ankle_true).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_hip_loss = hip_loss / len(val_dataset)
    avg_knee_loss = knee_loss / len(val_dataset)
    avg_ankle_loss = ankle_loss / len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f}")
    print(f"    Hip MSE: {avg_hip_loss:.2f} | Knee MSE: {avg_knee_loss:.2f} | Ankle MSE: {avg_ankle_loss:.2f}")

# === SAVE MODEL === #
model_path = os.path.join(base_path, "llr_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
