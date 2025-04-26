# File: train_llr_model_v3_6.py
"""
Training v3_6:
- Pixel-space SmoothL1Loss
- Extended epochs to 50
- Lower LR to 1e-4
- CosineAnnealingLR for smooth decay
- Save best by validation only (no early stopping)
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from llr_dataset_v3_6 import LLRDatasetV3_6
from llr_cnn_model_v3_6 import LLRLandmarkCNN_v3_6

# CONFIG
BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
SAVE = os.path.join(BASE, 'v3_6', 'llr_model_v3_6_best.pth')
W, H = 192, 640
EPOCHS, BATCH, LR = 50, 4, 1e-4
SEED = 42
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# reproducibility & device
torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda': torch.cuda.manual_seed_all(SEED)

# transforms
t_train = transforms.Compose([transforms.Resize((H,W)), transforms.ToTensor()])
t_val   = transforms.Compose([transforms.Resize((H,W)), transforms.ToTensor()])

# datasets & split
full = LLRDatasetV3_6(EXCEL, IMDIR, transform=None)
train_ds = LLRDatasetV3_6(EXCEL, IMDIR, transform=t_train, augment=True)
val_ds   = LLRDatasetV3_6(EXCEL, IMDIR, transform=t_val, augment=False)
keys = full.sample_keys
random.shuffle(keys)
split = int(0.8 * len(keys))
train_ds.sample_keys = keys[:split]
val_ds.sample_keys   = keys[split:]

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH)

# model, loss, optimizer, scheduler
model = LLRLandmarkCNN_v3_6().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val = float('inf')
for ep in range(1, EPOCHS+1):
    # train
    model.train(); train_loss = 0.0
    pixel_mse_train = [0.0]*6
    for imgs, coords in train_loader:
        imgs, coords = imgs.to(device), coords.to(device)
        preds_norm = model(imgs)
        # denormalize
        preds = preds_norm.clone()
        coords_px = coords.clone()
        preds[:, ::2] *= W; preds[:,1::2] *= H
        coords_px[:, ::2] *= W; coords_px[:,1::2] *= H
        loss = criterion(preds, coords_px)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
        diff2 = (preds - coords_px)**2
        mse = diff2.view(-1,6,2).mean(dim=2)
        for i in range(6): pixel_mse_train[i] += mse[:,i].mean().item()
    train_loss /= len(train_loader)
    rmse_train = [(m/len(train_loader))**0.5 for m in pixel_mse_train]

    # validation
    model.eval(); val_loss = 0.0; pixel_mse_val = [0.0]*6
    with torch.no_grad():
        for imgs, coords in val_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            preds_norm = model(imgs)
            preds = preds_norm.clone(); coords_px = coords.clone()
            preds[:, ::2] *= W; preds[:,1::2] *= H
            coords_px[:, ::2] *= W; coords_px[:,1::2] *= H
            l = criterion(preds, coords_px)
            val_loss += l.item()
            diff2 = (preds - coords_px)**2
            mse = diff2.view(-1,6,2).mean(dim=2)
            for i in range(6): pixel_mse_val[i] += mse[:,i].mean().item()
    val_loss /= len(val_loader)
    rmse_val = [(m/len(val_loader))**0.5 for m in pixel_mse_val]

    scheduler.step()

    print(f"Ep {ep}/{EPOCHS} | Huber Tr: {train_loss:.2f} | Val: {val_loss:.2f}")
    for i,name in enumerate(LANDMARKS):
        print(f"  {name} RMSE px - Tr:{rmse_train[i]:.2f} Val:{rmse_val[i]:.2f}")

    if val_loss < best_val:
        torch.save(model.state_dict(), SAVE)
        best_val = val_loss
        print(" * New best saved")

print("Training complete.")

