# train_llr_model_v3_2.py
"""
Training script for v3_2 (refactored):
- Tracks and prints per-landmark MSE losses for training and validation
- Uses consistent normalized coordinates in [0,1]
- Splits dataset reproducibly
- Uses ReduceLROnPlateau scheduler
- Saves best model checkpoint
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from llr_dataset_v3_2 import LLRDatasetV3_2
from llr_cnn_model_v3_2 import LLRLandmarkCNN_v3_2

# === CONFIGURATION ===
BASE_PATH = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL_FILE = os.path.join(BASE_PATH, 'v3_1', 'outputs.xlsx')
IMAGE_DIR = os.path.join(BASE_PATH, 'v3_1', 'data_acquisition', 'raw_data')
MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'v3_2', 'llr_model_v3_2_best.pth')

ORIG_WIDTH = 256
ORIG_HEIGHT = 896
TARGET_WIDTH = 192
TARGET_HEIGHT = 640
NUM_EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
SEED = 42
landmark_names = ['RH', 'RK', 'RA', 'LH', 'LK', 'LA']

# === REPRODUCIBILITY ===
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
    transforms.Lambda(lambda img: img.convert('L')),
    transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 2.0)),
    transforms.ToTensor(),
])

# === DATASET & SPLIT ===
dataset = LLRDatasetV3_2(EXCEL_FILE, IMAGE_DIR, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === MODEL, LOSS, OPTIMIZER, SCHEDULER ===
model = LLRLandmarkCNN_v3_2().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# === TRAIN LOOP ===
best_val_loss = float('inf')
for epoch in range(1, NUM_EPOCHS + 1):
    # Training phase
    model.train()
    running_loss = 0.0
    landmark_train_losses = [0.0] * len(landmark_names)

    for images, coords_norm in train_loader:
        images = images.to(device)
        coords_norm = coords_norm.to(device)

        preds = model(images)
        loss = criterion(preds, coords_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Per-landmark losses
        for i in range(len(landmark_names)):
            start = i * 2
            end = start + 2
            landmark_train_losses[i] += criterion(preds[:, start:end], coords_norm[:, start:end]).item()

    avg_train_loss = running_loss / len(train_loader)
    avg_landmark_train = [l / len(train_loader) for l in landmark_train_losses]

    # Validation phase
    model.eval()
    val_loss = 0.0
    landmark_val_losses = [0.0] * len(landmark_names)
    with torch.no_grad():
        for images, coords_norm in val_loader:
            images = images.to(device)
            coords_norm = coords_norm.to(device)

            preds = model(images)
            val_loss += criterion(preds, coords_norm).item()
            for i in range(len(landmark_names)):
                start = i * 2
                end = start + 2
                landmark_val_losses[i] += criterion(preds[:, start:end], coords_norm[:, start:end]).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_landmark_val = [l / len(val_loader) for l in landmark_val_losses]

    # Scheduler step
    scheduler.step(avg_val_loss)

    # Print summary
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    for i, name in enumerate(landmark_names):
        print(f"    {name} Train MSE: {avg_landmark_train[i]:.6f} | Val MSE: {avg_landmark_val[i]:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_val_loss = avg_val_loss
        print(f"  → New best model saved (Val Loss: {best_val_loss:.6f})")

print("Training complete.")
