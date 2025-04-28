# File: train_llr_model.py
"""
Training script for final ResNet18-based model with 70/15/15 train/val/test split:
- Pixel-space SmoothL1Loss on denormalized coords
- AdamW optimizer, CosineAnnealingLR, 50 epochs
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
BASE = os.path.dirname(os.path.abspath(__file__))
EXCEL = os.path.join(BASE, 'data_acquisition', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'data_acquisition', 'raw_data')
SAVE = os.path.join(BASE, 'llr_model_final.pth')
W, H = 192, 640
EPOCHS, BATCH, LR = 50, 4, 1e-4
SEED = 42
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# reproducibility & device
torch.manual_seed(SEED)
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda':
    torch.cuda.manual_seed_all(SEED)

# full dataset for splitting
full_ds = LLRDataset(EXCEL, IMDIR, transform=None)

# split sample_keys into 70/15/15
keys = full_ds.sample_keys[:] 
random.shuffle(keys)
n = len(keys)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_keys = keys[:n_train]
val_keys   = keys[n_train:n_train+n_val]
test_keys  = keys[n_train+n_val:]  # held-out

# prepare datasets
t_train = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
t_val   = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])

train_ds = LLRDataset(EXCEL, IMDIR, transform=t_train, augment=True)
val_ds   = LLRDataset(EXCEL, IMDIR, transform=t_val,   augment=False)
# assign our splits
train_ds.sample_keys = train_keys
val_ds.sample_keys   = val_keys

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH)

# model, loss, optimizer, scheduler
model     = LLRLandmarkCNN().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val = float('inf')
for ep in range(1, EPOCHS+1):
    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    pixel_mse_train = [0.0]*6

    for imgs, coords in train_loader:
        imgs, coords = imgs.to(device), coords.to(device)
        preds_norm = model(imgs)

        # denormalize out-of-place
        scale = torch.tensor([W, H] * 6, device=preds_norm.device, dtype=preds_norm.dtype).unsqueeze(0)
        preds    = preds_norm * scale
        coords_px = coords * scale

        loss = criterion(preds, coords_px)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        diff2 = (preds - coords_px) ** 2
        mse = diff2.view(-1,6,2).mean(dim=2)
        for i in range(6):
            pixel_mse_train[i] += mse[:,i].mean().item()

    train_loss /= len(train_loader)
    rmse_train = [(m / len(train_loader))**0.5 for m in pixel_mse_train]

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0
    pixel_mse_val = [0.0]*6

    with torch.no_grad():
        for imgs, coords in val_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            preds_norm = model(imgs)

            scale = torch.tensor([W, H] * 6, device=preds_norm.device, dtype=preds_norm.dtype).unsqueeze(0)
            preds    = preds_norm * scale
            coords_px = coords * scale

            l = criterion(preds, coords_px)
            val_loss += l.item()

            diff2 = (preds - coords_px) ** 2
            mse = diff2.view(-1,6,2).mean(dim=2)
            for i in range(6):
                pixel_mse_val[i] += mse[:,i].mean().item()

    val_loss /= len(val_loader)
    rmse_val = [(m / len(val_loader))**0.5 for m in pixel_mse_val]

    scheduler.step()

    print(f"Ep {ep}/{EPOCHS} | Huber Tr:{train_loss:.2f} | Val:{val_loss:.2f}")
    for i,name in enumerate(LANDMARKS):
        print(f"  {name} RMSE px - Tr:{rmse_train[i]:.2f} Val:{rmse_val[i]:.2f}")

    if val_loss < best_val:
        torch.save(model.state_dict(), SAVE)
        best_val = val_loss
        print(" * New best saved")

print("Training complete.")

