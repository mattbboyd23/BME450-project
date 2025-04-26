"""
llr_dataset_v3_1.py

Final clean dataset loader:
- Groups 6 landmarks by PatientID and Sample
- Loads image using correct "sampleX-YYYYYYY-resized.jpg" pattern
- Rescales X, Y coordinates properly
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

        # Group by Sample number (NOT just PatientID)
        self.grouped = self.data.groupby('Sample')

        # Store list of sample numbers for indexing
        self.sample_numbers = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.sample_numbers)

    def __getitem__(self, idx):
        sample_number = self.sample_numbers[idx]
        group = self.grouped.get_group(sample_number)

        # Patient ID (same within the group)
        patient_id = group.iloc[0]['PatientID']

        # Build correct image filename: sampleX-YYYYYYY-resized.jpg
        image_filename = f"sample{sample_number}-{patient_id}-resized.jpg"
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
