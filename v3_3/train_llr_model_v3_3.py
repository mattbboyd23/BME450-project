# File: train_llr_model_v3_3.py
"""
Training script v3_3:
- Weighted MSE to emphasize hip landmarks
- Early stopping + ReduceLROnPlateau
- Per-landmark MSE logging
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from llr_dataset_v3_3 import LLRDatasetV3_3
from llr_cnn_model_v3_3 import LLRLandmarkCNN_v3_3

# === CONFIG ===
BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IM_DIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
SAVE_PATH = os.path.join(BASE, 'v3_3', 'llr_model_v3_3_best.pth')

TARGET_W, TARGET_H = 192, 640
NUM_EPOCHS = 25
BATCH = 4
LR = 1e-3
SEED = 42
PATIENCE_STOP = 7
landmarks = ['RH','RK','RA','LH','LK','LA']

# device & seed
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda': torch.cuda.manual_seed_all(SEED)

# transforms
train_transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 2.0)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.ToTensor(),
])

# dataset & split
full_ds = LLRDatasetV3_3(EXCEL, IM_DIR, transform=None)
train_ds = LLRDatasetV3_3(EXCEL, IM_DIR, transform=train_transform, augment=True)
val_ds   = LLRDatasetV3_3(EXCEL, IM_DIR, transform=val_transform, augment=False)
# ensure non-overlapping: split keys first
keys = full_ds.sample_keys
import random; random.seed(SEED)
train_keys = set(random.sample(keys, int(0.8*len(keys))))
# repartition
train_ds.sample_keys = [k for k in keys if k in train_keys]
val_ds.sample_keys   = [k for k in keys if k not in train_keys]

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH)

# model
model = LLRLandmarkCNN_v3_3().to(device)
# weighted MSE: higher weight for hips
weights = torch.tensor([1.5,1.0,1.0,1.5,1.0,1.0], device=device)
def weighted_mse(pred, tgt):
    diff2 = (pred - tgt)**2
    diff2 = diff2.view(-1,6,2).mean(dim=2)  # [batch,6]
    return (diff2 * weights).mean()
mse = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=LR)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, verbose=True)

# training loop
best_val = float('inf'); stale=0
for ep in range(1, NUM_EPOCHS+1):
    # train
    model.train(); train_loss=0
    t_losses = [0]*6
    for imgs, coords in train_loader:
        imgs, coords = imgs.to(device), coords.to(device)
        preds = model(imgs)
        loss = weighted_mse(preds, coords)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss += loss.item()
        # logging unweighted
        for i in range(6):
            t_losses[i] += mse(preds[:,2*i:2*i+2], coords[:,2*i:2*i+2]).item()
    train_loss /= len(train_loader)
    t_losses = [l/len(train_loader) for l in t_losses]
    # val
    model.eval(); val_loss=0; v_losses=[0]*6
    with torch.no_grad():
        for imgs, coords in val_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            preds = model(imgs)
            v_loss = weighted_mse(preds, coords)
            val_loss += v_loss.item()
            for i in range(6):
                v_losses[i] += mse(preds[:,2*i:2*i+2], coords[:,2*i:2*i+2]).item()
    val_loss /= len(val_loader)
    v_losses = [l/len(val_loader) for l in v_losses]
    sched.step(val_loss)

    # print
    print(f"Ep {ep}/{NUM_EPOCHS} | Tr: {train_loss:.4f} | Val: {val_loss:.4f}")
    for i,name in enumerate(landmarks):
        print(f"    {name} Tr: {t_losses[i]:.4f} | Val: {v_losses[i]:.4f}")

    # early stop & save
    if val_loss < best_val:
        torch.save(model.state_dict(), SAVE_PATH); best_val=val_loss; stale=0
        print("  * New best saved")
    else:
        stale +=1
        if stale>=PATIENCE_STOP:
            print("Early stopping triggered")
            break

print("Done.")
