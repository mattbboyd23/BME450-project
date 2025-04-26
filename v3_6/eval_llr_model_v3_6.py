# File: eval_llr_model_v3_6.py
"""
Eval v3_6:
- Loads best v3_6 checkpoint
- Saves overlay PNGs and prints pixel errors
- Uses larger potent yellow dots for ground truth and potent blue dots for predictions
"""
import os
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from llr_dataset_v3_6 import LLRDatasetV3_6
from llr_cnn_model_v3_6 import LLRLandmarkCNN_v3_6

# Configuration
BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
CKPT = os.path.join(BASE, 'v3_6', 'llr_model_v3_6_best.pth')
W, H, N, SEED = 192, 640, 5, 42

# Set seeds for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare dataset and model
transform = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
ds = LLRDatasetV3_6(EXCEL, IMDIR, transform=transform, augment=False)
model = LLRLandmarkCNN_v3_6().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# Directory to save plots
outdir = os.path.dirname(os.path.abspath(__file__))
# Randomly select samples
indices = random.sample(range(len(ds)), N)

# Plot settings
gt_color = 'yellow'
pred_color = 'blue'
marker_size = 100  # larger dot size

for idx in indices:
    img_t, coords = ds[idx]
    img = img_t.squeeze(0).cpu().numpy()
    gt = coords.clone()
    with torch.no_grad():
        pred = model(img_t.unsqueeze(0).to(device)).cpu()[0]
    # Convert to pixel coordinates
    gt_px = gt.clone()
    pred_px = pred.clone()
    gt_px[::2] *= W; gt_px[1::2] *= H
    pred_px[::2] *= W; pred_px[1::2] *= H
    # Compute radial errors for printout
    diffs = (pred_px - gt_px).view(6, 2)
    errs = torch.sqrt((diffs**2).sum(dim=1) / 2)

    # Create plot
    plt.figure(figsize=(4, 12))
    plt.imshow(img, cmap='gray')
    # Ground truth points
    plt.scatter(gt_px[::2], gt_px[1::2], c=gt_color, marker='o', s=marker_size, label='Ground Truth', edgecolor='k')
    # Prediction points
    plt.scatter(pred_px[::2], pred_px[1::2], c=pred_color, marker='o', s=marker_size, label='Prediction', edgecolor='k')
    plt.title(f'v3_6 Sample {idx} | Mean Err: {errs.mean():.1f} px')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()

    # Save figure
    path = os.path.join(outdir, f'eval_v3_6_{idx}.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved {path} | Pixel Errors per Landmark: {errs.tolist()}")
