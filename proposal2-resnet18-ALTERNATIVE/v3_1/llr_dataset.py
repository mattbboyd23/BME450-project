"""
llr_dataset_v3_1.py

Clean dataset loader:
- Groups 6 landmarks by PatientID
- Loads images based on PatientID
- Rescales X, Y coordinates based on resized image dimensions
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class LLRDataset(Dataset):
    def __init__(self, excel_file, image_dir, transform=None):
        self.data = pd.read_excel(excel_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_width = 192
        self.target_height = 640

        # Group by PatientID
        self.grouped = self.data.groupby('PatientID')

        # Store list of patient IDs for indexing
        self.patient_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        group = self.grouped.get_group(patient_id)

        # Load image
        image_filename = f"{patient_id}-resized.jpg"
        img_path = os.path.join(self.image_dir, image_filename)
        img = Image.open(img_path).convert('L')  # grayscale

        # Resize image
        img = img.resize((self.target_width, self.target_height))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())),
                               dtype=torch.uint8)
            img = img.view(self.target_height, self.target_width).unsqueeze(0).float() / 255.0

        # Build coordinate array
        label_map = {'RH': 0, 'RK': 1, 'RA': 2, 'LH': 3, 'LK': 4, 'LA': 5}
        coords = torch.zeros(12, dtype=torch.float32)

        for _, row in group.iterrows():
            label = row['Label']
            x = row['X'] * (self.target_width / 256)
            y = row['Y'] * (self.target_height / 896)

            idx = label_map[label]
            coords[idx * 2] = x
            coords[idx * 2 + 1] = y

        return img, coords
