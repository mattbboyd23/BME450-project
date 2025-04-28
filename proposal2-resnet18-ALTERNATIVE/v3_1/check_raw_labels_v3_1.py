"""
check_raw_labels_v3_1.py

Loads resized image.
Plots raw (x, y) landmarks directly from outputs.xlsx without model.
Used to visually verify if raw label coordinates are correct.
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

# === PATHS ===
base_path = r'C:\Users\Mattb\Downloads\PURDUE\SPRING2025\BME450\PROJECT\BME450-project'
excel_file = os.path.join(base_path, 'v3_1', 'outputs.xlsx')
image_dir = os.path.join(base_path, 'v3_1', 'data_acquisition', 'raw_data')

# === SETTINGS ===
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 640

# Load outputs.xlsx
data = pd.read_excel(excel_file)

# Group by Sample
grouped = data.groupby('Sample')

# Pick which sample to check
sample_number = 23  # change this to any sample you want to check!

group = grouped.get_group(sample_number)
patient_id = group.iloc[0]['PatientID']

# Build filename
image_filename = f"sample{sample_number}-{patient_id}-resized.jpg"
img_path = os.path.join(image_dir, image_filename)

# Load and resize image
img = Image.open(img_path).convert('L')
img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

# Prepare figure
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(img, cmap='gray')

# Plot landmarks
label_colors = {
    'RH': 'red',
    'RK': 'blue',
    'RA': 'green',
    'LH': 'orange',
    'LK': 'purple',
    'LA': 'cyan'
}

for _, row in group.iterrows():
    label = row['Label']
    x = row['X'] * (IMAGE_WIDTH / 256)
    y = row['Y'] * (IMAGE_HEIGHT / 896)
    ax.plot(x, y, 'o', color=label_colors.get(label, 'yellow'), markersize=6)
    ax.text(x + 3, y, label, color=label_colors.get(label, 'yellow'), fontsize=8, fontweight='bold')

ax.set_title(f'Raw Labels Overlay: Sample {sample_number}')
ax.axis('off')

save_path = os.path.join(base_path, 'v3_1', f'check_raw_labels_sample{sample_number}.png')
plt.savefig(save_path)
print(f"Saved raw label check to: {save_path}")
