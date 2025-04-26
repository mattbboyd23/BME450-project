# eval_llr_model_v3_3.py
"""
Evaluation script for v3_3:
- Loads the best v3_3 checkpoint
- Runs inference on random samples
- Saves overlaid ground-truth vs. predicted landmarks as PNGs
"""
import os
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from llr_dataset_v3_3 import LLRDatasetV3_3
from llr_cnn_model_v3_3 import LLRLandmarkCNN_v3_3

# === CONFIGURATION ===
BASE_PATH = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL_FILE = os.path.join(BASE_PATH, 'v3_1', 'outputs.xlsx')
IMAGE_DIR  = os.path.join(BASE_PATH, 'v3_1', 'data_acquisition', 'raw_data')
MODEL_PATH = os.path.join(BASE_PATH, 'v3_3', 'llr_model_v3_3_best.pth')
TARGET_W, TARGET_H = 192, 640
NUM_SAMPLES = 5
SEED = 42

# === SETUP ===
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === DATASET & TRANSFORM ===
eval_transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),
    transforms.ToTensor(),
])
dataset = LLRDatasetV3_3(EXCEL_FILE, IMAGE_DIR, transform=eval_transform, augment=False)

# === MODEL LOADING ===
model = LLRLandmarkCNN_v3_3().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === SAMPLE AND EVALUATE ===
indices = random.sample(range(len(dataset)), NUM_SAMPLES)
script_dir = os.path.dirname(os.path.abspath(__file__))

for idx in indices:
    img_tensor, coords_norm = dataset[idx]
    img = img_tensor.squeeze(0).cpu().numpy()
    true_norm = coords_norm.cpu().numpy()

    with torch.no_grad():
        pred_norm = model(img_tensor.unsqueeze(0).to(device)).cpu().numpy().squeeze(0)

    # Denormalize
    true_x = true_norm[0::2] * TARGET_W
    true_y = true_norm[1::2] * TARGET_H
    pred_x = pred_norm[0::2] * TARGET_W
    pred_y = pred_norm[1::2] * TARGET_H

    # Plot
    plt.figure(figsize=(4, 12))
    plt.imshow(img, cmap='gray')
    plt.scatter(true_x, true_y, marker='o', label='GT', edgecolor='w')
    plt.scatter(pred_x, pred_y, marker='x', label='Pred', edgecolor='k')
    plt.title(f'v3_3 Eval Sample {idx}')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(script_dir, f'eval_v3_3_sample_{idx}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved v3_3 eval plot: {out_path}")
