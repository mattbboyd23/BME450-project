#File: eval_llr_model_v3_4.py
"""
Eval v3_4:
- Load v3_4 checkpoint, sample random images, overlay preds vs GT, save PNGs
"""
import os
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from llr_dataset_v3_4 import LLRDatasetV3_4
from llr_cnn_model_v3_4 import LLRLandmarkCNN_v3_4

BASE = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL = os.path.join(BASE, 'v3_1', 'outputs.xlsx')
IMDIR = os.path.join(BASE, 'v3_1', 'data_acquisition', 'raw_data')
CKPT = os.path.join(BASE, 'v3_4', 'llr_model_v3_4_best.pth')
W, H = 192,640; N=5; SEED=42

torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([transforms.Resize((H,W)), transforms.ToTensor()])
ds = LLRDatasetV3_4(EXCEL, IMDIR, transform=trans, augment=False)
model = LLRLandmarkCNN_v3_4().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()

outdir = os.path.dirname(os.path.abspath(__file__))
idxs = random.sample(range(len(ds)), N)
for idx in idxs:
    img_t, gt = ds[idx]
    img = img_t.squeeze(0).cpu().numpy(); gt_n=gt.cpu().numpy()
    with torch.no_grad(): pr = model(img_t.unsqueeze(0).to(device)).cpu().numpy()[0]
    tx,ty = gt_n[0::2]*W, gt_n[1::2]*H
    px,py = pr[0::2]*W, pr[1::2]*H
    plt.figure(figsize=(4,12)); plt.imshow(img, cmap='gray')
    plt.scatter(tx,ty, marker='o', label='GT', edgecolor='w')
    plt.scatter(px,py, marker='x', label='Pred', edgecolor='k')
    plt.title(f'v3_4 Sample {idx}'); plt.axis('off'); plt.legend(); plt.tight_layout()
    p = os.path.join(outdir, f'eval_v3_4_{idx}.png'); plt.savefig(p); plt.close()
    print('Saved', p)
