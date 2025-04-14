"""
llr_dataset.py

Custom PyTorch Dataset for Long-Leg Radiograph (LLR) X-ray images and anatomical landmark detection.
Each sample includes six keypoints:
    - RH: Right Hip
    - RK: Right Knee
    - RA: Right Ankle
    - LH: Left Hip
    - LK: Left Knee
    - LA: Left Ankle

This dataset reads:
    - Images from a specified directory
    - Ground truth pixel coordinates from 'outputs.xlsx'

Image filenames are reconstructed automatically using:
    sample{Sample}-{PatientID}-resized.jpg

Returns:
    - Grayscale image tensor: [1, H, W]
    - Landmark coordinate tensor: [12]
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LLRDataset(Dataset):
    def __init__(self, excel_file, image_dir, transform=None):
        """
        Args:
            excel_file (str): Path to the Excel file with annotations.
            image_dir (str): Directory where the image files are stored.
            transform (callable, optional): Optional transforms to be applied on images.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Labels to include and keep consistent ordering
        target_labels = ['RH', 'RK', 'RA', 'LH', 'LK', 'LA']

        # Load and filter data
        df = pd.read_excel(excel_file)
        df = df[df['Label'].isin(target_labels)]

        self.samples = []

        # Group rows by sample number
        for sample_id, group in df.groupby('Sample'):
            # Ensure label order is consistent
            group = group.set_index('Label').loc[target_labels].reset_index()

            # Flatten coordinates into 1D array [x1, y1, ..., x6, y6]
            coords = group[['X', 'Y']].values.flatten().astype('float32')

            # Reconstruct filename from sample number and patient ID
            sample_num = group['Sample'].iloc[0]
            patient_id = group['PatientID'].iloc[0]
            filename = f"sample{sample_num}-{patient_id}-resized.jpg"

            # Add to internal sample list
            self.samples.append({
                'filename': filename,
                'coords': coords
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Grayscale image tensor [1, H, W]
            coords (Tensor): Tensor of shape [12] with landmark coordinates
        """
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample['filename'])

        # Load image and convert to grayscale
        image = Image.open(img_path).convert("L")

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        # Format target coordinates
        coords = torch.tensor(sample['coords'], dtype=torch.float32)

        return image, coords
