# eval_llr_model_v3_2.py
"""
Evaluation script for v3_2:
- Loads the best checkpoint
- Runs inference on a set of random samples
- Saves plots of ground-truth vs. predicted landmarks overlaid on images as PNGs
"""

import os
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from llr_dataset_v3_2 import LLRDatasetV3_2
from llr_cnn_model_v3_2 import LLRLandmarkCNN_v3_2

# === CONFIGURATION ===
BASE_PATH = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
EXCEL_FILE = os.path.join(BASE_PATH, 'v3_1', 'outputs.xlsx')
IMAGE_DIR = os.path.join(BASE_PATH, 'v3_1', 'data_acquisition', 'raw_data')
MODEL_PATH = os.path.join(BASE_PATH, 'v3_2', 'llr_model_v3_2_best.pth')
TARGET_WIDTH = 192
TARGET_HEIGHT = 640
NUM_SAMPLES = 5
SEED = 42

# === REPRODUCIBILITY ===
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === DATASET ===
eval_transform = transforms.Compose([
    transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
    transforms.Lambda(lambda img: img.convert('L')),
    transforms.ToTensor(),
])
dataset = LLRDatasetV3_2(EXCEL_FILE, IMAGE_DIR, transform=eval_transform)

# === MODEL LOAD ===
model = LLRLandmarkCNN_v3_2().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === SAMPLING ===
indices = random.sample(range(len(dataset)), NUM_SAMPLES)

# Directory to save plots (script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))

for idx in indices:
    img_tensor, coords_norm = dataset[idx]
    img = img_tensor.squeeze(0).cpu().numpy()
    true_norm = coords_norm.cpu().numpy()

    # Inference
    with torch.no_grad():
        pred_norm = model(img_tensor.unsqueeze(0).to(device)).cpu().numpy().squeeze(0)

    # Denormalize to resized image pixel coords
    true_x = true_norm[0::2] * TARGET_WIDTH
    true_y = true_norm[1::2] * TARGET_HEIGHT
    pred_x = pred_norm[0::2] * TARGET_WIDTH
    pred_y = pred_norm[1::2] * TARGET_HEIGHT

    # Plot
    plt.figure(figsize=(4, 12))
    plt.imshow(img, cmap='gray')
    plt.scatter(true_x, true_y, c='g', marker='o', label='Ground Truth')
    plt.scatter(pred_x, pred_y, c='r', marker='x', label='Prediction')
    plt.title(f'Sample Index: {idx}')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    # Save figure as PNG
    save_path = os.path.join(script_dir, f'eval_sample_{idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved evaluation plot for sample {idx} to {save_path}")