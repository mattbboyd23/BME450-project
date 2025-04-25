"""
predict_and_visualize.py

Loads trained model and visualizes predicted vs. ground truth landmarks
for one LLR X-ray. Ground truth = yellow, Prediction = blue.
"""

import torch
import matplotlib.pyplot as plt
from llr_cnn_model import LLRLandmarkCNN
from llr_dataset import LLRDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# === Set paths based on your actual structure === #
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
model_path = os.path.join(base_path, 'v1', 'llr_model.pth')
excel_file = os.path.join(base_path, 'v1', 'outputs.xlsx')
image_dir = os.path.join(base_path, 'v1', 'data_acquisition', 'raw_data')  # updated

# === Image transformation (same as training) === #
transform = transforms.Compose([
    transforms.Resize((896, 256)),
    transforms.ToTensor()
])

# === Load trained model === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LLRLandmarkCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load dataset === #
dataset = LLRDataset(excel_file=excel_file, image_dir=image_dir, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# === Get one prediction === #
for image, true_coords in loader:
    image = image.to(device)
    true_coords = true_coords[0].numpy().reshape(6, 2)

    with torch.no_grad():
        pred_coords = model(image).cpu().numpy().reshape(6, 2)

    # Convert image to 2D NumPy
    img_np = image.cpu().squeeze().numpy()

    # Plot
    plt.imshow(img_np, cmap='gray')
    plt.title("Yellow = Ground Truth | Blue = Prediction")

    # Plot landmarks
    for (x, y) in true_coords:
        plt.plot(x, y, 'yo', markersize=4)  # Ground truth (yellow)
    for (x, y) in pred_coords:
        plt.plot(x, y, 'bo', markersize=4)  # Prediction (blue)

    # Save and show
    save_path = os.path.join(base_path, 'v1', 'sample_prediction.png')
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    plt.show()
    break

