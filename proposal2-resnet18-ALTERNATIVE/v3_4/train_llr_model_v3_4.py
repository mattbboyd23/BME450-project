# File: train_llr_model_v3_4.py
"""
Training v3_4:
- AdamW with weight decay
- Weighted MSE emphasizing hips (2×)
- Early stopping + ReduceLROnPlateau
- Per-landmark MSE logging
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from llr_dataset_v3_4 import LLRDatasetV3_4
from llr_cnn_model_v3_4 import LLRLandmarkCNN_v3_4

# CONFIG
BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
SAVE = os.path.join(BASE, 'v3_4', 'llr_model_v3_4_best.pth')
W, H = 192, 640
EPOCHS = 25; BATCH = 4; LR = 1e-3; SEED = 42; PATIENCE = 5
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# seeds & device
torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda': torch.cuda.manual_seed_all(SEED)

# transforms
t_train = transforms.Compose([
    transforms.Resize((H,W)),
    transforms.ToTensor(),
])
t_val = transforms.Compose([
    transforms.Resize((H,W)),
    transforms.ToTensor(),
])

# datasets & split
full = LLRDatasetV3_4(EXCEL, IMDIR, transform=None)
train_ds = LLRDatasetV3_4(EXCEL, IMDIR, transform=t_train, augment=True)
val_ds   = LLRDatasetV3_4(EXCEL, IMDIR, transform=t_val, augment=False)
keys = full.sample_keys
random.shuffle(keys)
split = int(0.8*len(keys))
train_ds.sample_keys = keys[:split]
val_ds.sample_keys   = keys[split:]

t_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
v_loader = DataLoader(val_ds, batch_size=BATCH)

# model, loss, opt, sched
model = LLRLandmarkCNN_v3_4().to(device)
# weighted MSE (hips ×2)
weights = torch.tensor([2.0,1.0,1.0,2.0,1.0,1.0], device=device)
def weighted_mse(p,t):
    d2 = (p-t)**2
    lm = d2.view(-1,6,2).mean(dim=2)  # [batch,6]
    return (lm * weights).mean()
mse = nn.MSELoss()
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

# training loop
best_val = float('inf'); wait = 0
for ep in range(1, EPOCHS+1):
    # train
    model.train(); t_loss=0; tl=[0]*6
    for imgs, coords in t_loader:
        imgs, coords = imgs.to(device), coords.to(device)
        preds = model(imgs)
        loss = weighted_mse(preds, coords)
        opt.zero_grad(); loss.backward(); opt.step()
        t_loss += loss.item()
        for i in range(6): tl[i] += mse(preds[:,2*i:2*i+2], coords[:,2*i:2*i+2]).item()
    t_loss /= len(t_loader); tl = [x/len(t_loader) for x in tl]
    # val
    model.eval(); v_loss=0; vl=[0]*6
    with torch.no_grad():
        for imgs, coords in v_loader:
            imgs, coords = imgs.to(device), coords.to(device)
            preds = model(imgs)
            v = weighted_mse(preds, coords)
            v_loss += v.item()
            for i in range(6): vl[i] += mse(preds[:,2*i:2*i+2], coords[:,2*i:2*i+2]).item()
    v_loss /= len(v_loader); vl = [x/len(v_loader) for x in vl]
    sched.step(v_loss)
    print(f"Ep {ep}/{EPOCHS} | Tr: {t_loss:.4f} | Val: {v_loss:.4f}")
    for i,name in enumerate(LANDMARKS): print(f"  {name} Tr:{tl[i]:.4f} Val:{vl[i]:.4f}")
    # early stop & save
    if v_loss < best_val:
        torch.save(model.state_dict(), SAVE); best_val=v_loss; wait=0; print(' *New best saved')
    else:
        wait +=1
        if wait >= PATIENCE:
            print('Early stopping'); break
print('Done')
