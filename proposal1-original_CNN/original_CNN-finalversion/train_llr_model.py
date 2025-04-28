# File: train_llr_model.py

"""
Training script for final original CNN with 70/15/15 split:
- Pixel-space SmoothL1Loss on denormalized coords
- AdamW optimizer, CosineAnnealingLR, 50 epochs
- Reproducible 70% train, 15% val, 15% test split (test reserved for eval)
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from llr_dataset import LLRDataset
from llr_cnn_model import LLRLandmarkCNN

# Paths relative to this script
BASE    = os.path.dirname(os.path.abspath(__file__))
EXCEL   = os.path.join(BASE, 'data_acquisition', 'outputs.xlsx')
IMDIR   = os.path.join(BASE, 'data_acquisition', 'raw_data')
SAVE    = os.path.join(BASE, 'llr_model_final.pth')
W, H    = 192, 640
EPOCHS, BATCH, LR = 50, 4, 1e-4
SEED    = 42
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# Reproducibility & device
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda':
    torch.cuda.manual_seed_all(SEED)

# Prepare full dataset and split 70/15/15
full_ds = LLRDataset(EXCEL, IMDIR, transform=None, augment=False)
keys = full_ds.sample_keys.copy()
random.shuffle(keys)
n = len(keys)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_keys = keys[:n_train]
val_keys   = keys[n_train:n_train+n_val]
# test_keys = keys[n_train+n_val:]  # reserved for eval

# Create train and val datasets
transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
train_ds = LLRDataset(EXCEL, IMDIR, transform=transform, augment=True)
val_ds   = LLRDataset(EXCEL, IMDIR, transform=transform, augment=False)
train_ds.sample_keys = train_keys
val_ds.sample_keys   = val_keys

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH)

# Model, loss, optimizer, scheduler
model     = LLRLandmarkCNN().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val = float('inf')
os.makedirs(BASE, exist_ok=True)

for ep in range(1, EPOCHS+1):
    # Training
    model.train()
    total_loss = 0.0
    pixel_sq_err = [0.0]*6
    count = 0

    for imgs, coords in train_loader:
        bs = imgs.size(0)
        count += bs
        imgs, coords = imgs.to(device), coords.to(device)
        preds_norm = model(imgs)

        # Denormalize to pixel space
        preds = preds_norm.clone()
        coords_px = coords.clone()
        preds[:, ::2] *= W; preds[:, 1::2] *= H
        coords_px[:, ::2] *= W; coords_px[:, 1::2] *= H

        loss = criterion(preds, coords_px)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

        # accumulate squared error per landmark
        dx = preds[:, 0::2] - coords_px[:, 0::2]
        dy = preds[:, 1::2] - coords_px[:, 1::2]
        mse = ((dx**2 + dy**2)/2).mean(dim=0)
        for i in range(6):
            pixel_sq_err[i] += mse[i].item() * bs

    avg_train_loss = total_loss / len(train_loader)
    rmse_train = [ (e/count)**0.5 for e in pixel_sq_err ]

    # Validation
    model.eval()
    total_val = 0.0
    pixel_sq_err_val = [0.0]*6
    vcount = 0
    with torch.no_grad():
        for imgs, coords in val_loader:
            bs = imgs.size(0)
            vcount += bs
            imgs, coords = imgs.to(device), coords.to(device)
            preds_norm = model(imgs)

            preds = preds_norm.clone()
            coords_px = coords.clone()
            preds[:, ::2] *= W; preds[:, 1::2] *= H
            coords_px[:, ::2] *= W; coords_px[:, 1::2] *= H

            l = criterion(preds, coords_px)
            total_val += l.item()

            dx = preds[:, 0::2] - coords_px[:, 0::2]
            dy = preds[:, 1::2] - coords_px[:, 1::2]
            mse = ((dx**2 + dy**2)/2).mean(dim=0)
            for i in range(6):
                pixel_sq_err_val[i] += mse[i].item() * bs

    avg_val_loss = total_val / len(val_loader)
    rmse_val = [ (e/vcount)**0.5 for e in pixel_sq_err_val ]

    scheduler.step()

    print(f"Ep {ep}/{EPOCHS} | Huber Tr:{avg_train_loss:.2f} | Val:{avg_val_loss:.2f}")
    for i, name in enumerate(LANDMARKS):
        print(f"  {name} RMSE px - Tr:{rmse_train[i]:.2f} Val:{rmse_val[i]:.2f}")

    if avg_val_loss < best_val:
        torch.save(model.state_dict(), SAVE)
        best_val = avg_val_loss
        print(" * New best saved")

print("Training complete.")
