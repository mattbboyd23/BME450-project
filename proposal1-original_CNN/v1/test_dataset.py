"""
test_dataset.py

This script tests the custom PyTorch Dataset (LLRDataset) for loading
long-leg radiograph (LLR) images and their corresponding 6 anatomical 
landmark coordinates (RH, RK, RA, LH, LK, LA).

Images are resized to 896x256 (portrait format). Landmarks are plotted
as yellow dots for visual verification, and saved as a PNG to the project folder.
"""

from llr_dataset import LLRDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

# Base path where images and outputs.xlsx are located
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT'

# Transform: Resize images and convert to grayscale tensor
transform = transforms.Compose([
    transforms.Resize((896, 256)),  # height x width (portrait mode)
    transforms.ToTensor()
])

# Load dataset
dataset = LLRDataset(
    excel_file=f'{base_path}\\outputs.xlsx',
    image_dir=base_path,
    transform=transform
)

# Load one random sample (change shuffle=False to always get the first sample)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Get and visualize one sample
for img, coords in loader:
    print("Image shape:", img.shape)   # [1, 1, 896, 256]
    print("Coords:", coords)           # [1, 12]

    # Convert tensor to NumPy format for visualization
    img_np = img.squeeze().numpy()         # shape: [896, 256]
    coords_np = coords.squeeze().numpy()   # shape: [12]
    points = coords_np.reshape(6, 2)       # [6, 2] (x, y) pairs

    # Plot image and yellow landmark points
    plt.imshow(img_np, cmap='gray')
    for (x, y) in points:
        plt.plot(x, y, 'yo', markersize=4)  # yellow dot, smaller for precision
    plt.title("Ground Truth Landmarks: RH, RK, RA, LH, LK, LA")

    # Save the plot
    save_path = os.path.join(base_path, "landmark_overlay.png")
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    break
