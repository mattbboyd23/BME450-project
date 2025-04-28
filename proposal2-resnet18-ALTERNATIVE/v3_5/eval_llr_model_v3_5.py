# File: eval_llr_model_v3_5.py
"""
Eval v3_5:
- Loads best v3_5 checkpoint
- Saves overlay PNGs and prints pixel errors
"""
import os
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from llr_dataset_v3_5 import LLRDatasetV3_5
from llr_cnn_model_v3_5 import LLRLandmarkCNN_v3_5

BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
CKPT = os.path.join(BASE, 'v3_5', 'llr_model_v3_5_best.pth')
W, H, N, SEED = 192, 640, 5, 42

random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((H,W)), transforms.ToTensor()])
ds = LLRDatasetV3_5(EXCEL, IMDIR, transform=transform, augment=False)
model = LLRLandmarkCNN_v3_5().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

mkdir = os.path.dirname(os.path.abspath(__file__))
indices = random.sample(range(len(ds)), N)
for idx in indices:
    img_t, coords = ds[idx]
    img = img_t.squeeze(0).cpu().numpy()
    gt = coords.clone()
    with torch.no_grad(): pred = model(img_t.unsqueeze(0).to(device)).cpu()[0]
    # pixel coords
    gt_px = gt.clone(); pred_px = pred.clone()
    gt_px[::2] *= W; gt_px[1::2] *= H
    pred_px[::2] *= W; pred_px[1::2] *= H
    # compute radial error
    diffs = (pred_px - gt_px).view(6,2)
    errs = torch.sqrt((diffs**2).sum(dim=1)/2)
    # plot
    plt.figure(figsize=(4,12)); plt.imshow(img, cmap='gray')
    plt.scatter(gt_px[::2], gt_px[1::2], c='g', marker='o', label='GT')
    plt.scatter(pred_px[::2], pred_px[1::2], c='r', marker='x', label='Pred')
    plt.title(f'v3_5 Sample {idx} | Mean Err:{errs.mean():.1f}px')
    plt.axis('off'); plt.legend(); plt.tight_layout()
    path = os.path.join(mkdir, f'eval_v3_5_{idx}.png')
    plt.savefig(path); plt.close()
    print(f"Saved {path} | Pixel Errors per LM: {errs.tolist()}")
