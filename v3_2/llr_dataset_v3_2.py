# llr_dataset_v3_2.py
"""
Dataset loader for v3_2 (refactored):
- Groups by (PatientID, Sample)
- Uses transform for all resizing
- Returns normalized coords in [0,1]
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class LLRDatasetV3_2(Dataset):
    ORIG_WIDTH = 256
    ORIG_HEIGHT = 896
    TARGET_WIDTH = 192
    TARGET_HEIGHT = 640

    def __init__(self, excel_file, image_dir, transform=None):
        self.data = pd.read_excel(excel_file)
        self.image_dir = image_dir
        self.transform = transform
        # group by both PatientID and Sample
        self.grouped = self.data.groupby(['PatientID', 'Sample'])
        self.sample_keys = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        patient_id, sample_number = self.sample_keys[idx]
        group = self.grouped.get_group((patient_id, sample_number))

        filename = f"sample{sample_number}-{patient_id}-resized.jpg"
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        else:
            img = img.convert('L')
            img = img.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT))
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        label_map = {'RH': 0, 'RK': 1, 'RA': 2, 'LH': 3, 'LK': 4, 'LA': 5}
        coords = torch.zeros(12, dtype=torch.float32)
        for _, row in group.iterrows():
            label = row['Label']
            x_norm = row['X'] / self.ORIG_WIDTH
            y_norm = row['Y'] / self.ORIG_HEIGHT
            idx2 = label_map[label]
            coords[idx2 * 2] = x_norm
            coords[idx2 * 2 + 1] = y_norm

        return img, coords
