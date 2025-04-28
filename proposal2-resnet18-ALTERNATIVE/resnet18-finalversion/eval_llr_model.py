# File: eval_llr_model.py
"""
Evaluation script on held-out test set (15 samples):
- Overlay yellow circles (GT) & blue circles (Pred)
- Saves output PNGs in this folder
- Computes & prints overall Huber loss and per-landmark RMSE on test set
"""
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from llr_dataset import LLRDataset
from llr_cnn_model import LLRLandmarkCNN

BASE = os.path.dirname(os.path.abspath(__file__))
EXCEL = os.path.join(BASE, 'data_acquisition', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'data_acquisition', 'raw_data')
CKPT  = os.path.join(BASE, 'llr_model_final.pth')
W, H, SEED = 192, 640, 42
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build the same 70/15/15 split to identify test_keys
full_ds = LLRDataset(EXCEL, IMDIR, transform=None)
keys = full_ds.sample_keys[:]
random.shuffle(keys)
n = len(keys)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
test_keys = keys[n_train+n_val:]  # final 15 held-out

# prepare test dataset
transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
test_ds = LLRDataset(EXCEL, IMDIR, transform=transform, augment=False)
test_ds.sample_keys = test_keys

# load model
model = LLRLandmarkCNN().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# loss containers
criterion = nn.SmoothL1Loss(reduction='mean')
total_huber = 0.0
pixel_sq_err = torch.zeros(6, dtype=torch.float64)

# evaluate each test sample
for idx in range(len(test_ds)):
    img_t, coords = test_ds[idx]
    img = img_t.squeeze(0).cpu().numpy()
    with torch.no_grad():
        pred_norm = model(img_t.unsqueeze(0).to(device)).cpu()[0]

    # denormalize out-of-place
    scale = torch.tensor([W, H] * 6, dtype=torch.float32)
    preds = pred_norm * scale
    coords_px = coords * scale

    # compute huber loss
    hub = criterion(preds, coords_px).item()
    total_huber += hub

    # accumulate squared error per landmark
    dx = preds[0::2] - coords_px[0::2]
    dy = preds[1::2] - coords_px[1::2]
    # RMSE per landmark = sqrt(mean(dx^2 + dy^2))
    pixel_sq_err += (dx**2 + dy**2)

    # plot & save
    plt.figure(figsize=(4,12))
    plt.imshow(img, cmap='gray')
    plt.scatter(coords_px[0::2], coords_px[1::2],
                c='yellow', marker='o', s=100, edgecolor='k', label='GT')
    plt.scatter(preds[0::2], preds[1::2],
                c='blue', marker='o', s=100, edgecolor='k', label='Pred')
    plt.title(f'Test Sample {idx} | Huber: {hub:.2f}')
    plt.axis('off'); plt.legend(); plt.tight_layout()

    outp = os.path.join(BASE, f'eval_test_{idx}.png')
    plt.savefig(outp); plt.close()
    print(f"Saved {outp}   Huber Loss: {hub:.2f}")

# final averages
n_test = len(test_ds)
avg_huber = total_huber / n_test
rmse_per_lm = torch.sqrt(pixel_sq_err / n_test)

print("\n=== Test Set Metrics ===")
print(f"Average Huber Loss: {avg_huber:.3f}")
for i, name in enumerate(LANDMARKS):
    print(f"  {name}  RMSE: {rmse_per_lm[i]:.2f} px")


