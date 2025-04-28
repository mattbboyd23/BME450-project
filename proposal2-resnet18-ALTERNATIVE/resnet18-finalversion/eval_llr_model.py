# File: eval_llr_model.py

"""
Evaluation script on the reserved 15% test set:
- Loads final model checkpoint
- Runs inference on the fixed 15 test samples
- Saves overlay plots and prints per-landmark RMSE, average RMSE, and average Huber loss
"""

import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from llr_dataset import LLRDataset
from llr_cnn_model import LLRLandmarkCNN

# Paths
BASE  = os.path.dirname(os.path.abspath(__file__))
EXCEL = os.path.join(BASE, 'data_acquisition', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'data_acquisition', 'raw_data')
CKPT  = os.path.join(BASE, 'llr_model_final.pth')
W, H  = 192, 640
SEED  = 42
LANDMARKS = ['RH','RK','RA','LH','LK','LA']

# Reproducibility & device
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recompute the same 70/15/15 split
full_ds = LLRDataset(EXCEL, IMDIR, transform=None, augment=False)
keys = full_ds.sample_keys.copy()
random.shuffle(keys)
n = len(keys)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
test_keys = keys[n_train+n_val:]  # last 15% reserved

# Build test dataset
transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
test_ds = LLRDataset(EXCEL, IMDIR, transform=transform, augment=False)
test_ds.sample_keys = test_keys

# Load model
model = LLRLandmarkCNN().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# Huber loss criterion
criterion = torch.nn.SmoothL1Loss()

# Make output dir
outdir = BASE
os.makedirs(outdir, exist_ok=True)

# Metrics accumulators
total_sq_err = torch.zeros(6)
total_huber = 0.0
count = 0

# Evaluate all test samples
for idx in range(len(test_ds)):
    img_t, coords = test_ds[idx]
    with torch.no_grad():
        pred_norm = model(img_t.unsqueeze(0).to(device)).cpu()[0]

    # Denormalize
    gt_px = coords.clone()
    pred_px = pred_norm.clone()
    gt_px[::2]   *= W; gt_px[1::2]   *= H
    pred_px[::2] *= W; pred_px[1::2] *= H

    # accumulate Huber loss (on all 12 coords)
    total_huber += criterion(pred_px, gt_px).item()

    # Compute per-landmark squared error
    diffs = torch.stack([pred_px[0::2] - gt_px[0::2],
                         pred_px[1::2] - gt_px[1::2]], dim=1)
    sq_err = (diffs**2).sum(dim=1) / 2  # mean of x²,y²
    total_sq_err += sq_err
    count += 1

    # Plot overlay
    img = img_t.squeeze(0).cpu().numpy()
    plt.figure(figsize=(4,12))
    plt.imshow(img, cmap='gray')
    plt.scatter(gt_px[0::2], gt_px[1::2], c='yellow', s=100, edgecolor='k', label='GT')
    plt.scatter(pred_px[0::2], pred_px[1::2], c='blue',   s=100, edgecolor='k', label='Pred')
    plt.title(f'Test Sample {idx}')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'test_eval_{idx}.png'))
    plt.close()

# Print final metrics
rmse = (total_sq_err / count).sqrt()
avg_huber = total_huber / count

print("=== Test Set Metrics ===")
print(f"Average Huber Loss: {avg_huber:.4f}")
print("Per-Landmark RMSE (px):")
for name, err in zip(LANDMARKS, rmse):
    print(f"  {name}: {err:.2f}")
print(f"Average RMSE: {rmse.mean():.2f}")
