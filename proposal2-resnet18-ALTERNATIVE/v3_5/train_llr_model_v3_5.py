# File: train_llr_model_v3_5.py
"""
Training v3_5:
- Pixel-space SmoothL1Loss (Huber) directly on resized coords
- Logs pixel RMSE per landmark
- Early stopping + ReduceLROnPlateau + AdamW
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from llr_dataset_v3_5 import LLRDatasetV3_5
from llr_cnn_model_v3_5 import LLRLandmarkCNN_v3_5

# CONFIGURATION
BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
SAVE = os.path.join(BASE, 'v3_5', 'llr_model_v3_5_best.pth')
W, H = 192, 640
EPOCHS, BATCH, LR = 25, 4, 5e-4
SEED, PATIENCE = 42, 10
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# Reproducibility & device
torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda': torch.cuda.manual_seed_all(SEED)

# Transforms
t_train = transforms.Compose([
    transforms.Resize((H,W)),
    transforms.ToTensor(),
])
t_val = transforms.Compose([
    transforms.Resize((H,W)),
    transforms.ToTensor(),
])

# Dataset split
full_ds = LLRDatasetV3_5(EXCEL, IMDIR, transform=None)
train_ds = LLRDatasetV3_5(EXCEL, IMDIR, transform=t_train, augment=True)
val_ds   = LLRDatasetV3_5(EXCEL, IMDIR, transform=t_val, augment=False)
keys = full_ds.sample_keys
random.shuffle(keys)
split = int(0.8 * len(keys))
train_ds.sample_keys = keys[:split]
val_ds.sample_keys   = keys[split:]

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH)

# Model, loss, optimizer, scheduler
model = LLRLandmarkCNN_v3_5().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Training loop
best_val = float('inf'); wait = 0
for ep in range(1, EPOCHS+1):
    # Training
    model.train(); t_loss=0
    pixel_mse_train = [0.0]*6
    for imgs, coords in train_loader:
        imgs, coords = imgs.to(device), coords.to(device)
        preds_norm = model(imgs)
        # Denormalize to pixels
        preds_px = preds_norm.clone()
        coords_px = coords.clone()
        preds_px[:, ::2] *= W
        preds_px[:, 1::2] *= H
        coords_px[:, ::2] *= W
        coords_px[:, 1::2] *= H
        # Pixel-space Huber loss
        loss = criterion(preds_px, coords_px)

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        t_loss += loss.item()
        # per-landmark pixel MSE
        diff2 = (preds_px - coords_px)**2
        batch_mse = diff2.view(-1,6,2).mean(dim=2)
        for i in range(6): pixel_mse_train[i] += batch_mse[:,i].mean().item()

    t_loss /= len(train_loader)
    pixel_rmse_train = [(m/len(train_loader))**0.5 for m in pixel_mse_train]

    # Validation
    model.eval(); v_loss=0
    pixel_mse_val = [0.0]*6
    with torch.no_grad():
        for imgs, coords in val_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            preds_norm = model(imgs)
            preds_px = preds_norm.clone(); coords_px = coords.clone()
            preds_px[:, ::2] *= W; preds_px[:, 1::2] *= H
            coords_px[:, ::2] *= W; coords_px[:, 1::2] *= H
            loss = criterion(preds_px, coords_px)
            v_loss += loss.item()
            diff2 = (preds_px - coords_px)**2
            batch_mse = diff2.view(-1,6,2).mean(dim=2)
            for i in range(6): pixel_mse_val[i] += batch_mse[:,i].mean().item()

    v_loss /= len(val_loader)
    pixel_rmse_val = [(m/len(val_loader))**0.5 for m in pixel_mse_val]

    scheduler.step(v_loss)

    # Logging
    print(f"Ep {ep}/{EPOCHS} | Pixel Huber Tr: {t_loss:.2f} | Val: {v_loss:.2f}")
    for i,name in enumerate(LANDMARKS):
        print(f"  {name} RMSE px - Tr:{pixel_rmse_train[i]:.2f} Val:{pixel_rmse_val[i]:.2f}")

    # Early stopping & save
    if v_loss < best_val:
        torch.save(model.state_dict(), SAVE)
        best_val = v_loss; wait=0
        print(" *New best saved*")
    else:
        wait +=1
        if wait >= PATIENCE:
            print("Early stopping triggered")
            break

print("Training complete.")
