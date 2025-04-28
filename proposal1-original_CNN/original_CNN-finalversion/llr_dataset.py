# File: llr_dataset.py
"""
Dataset for final original CNN pipeline:
- Clinically-safe augmentations: tiny translate/scale
- Normalizes coords to [0,1]
- Loads from data_acquisition/{outputs.xlsx, raw_data}
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class LLRDataset(Dataset):
    ORIG_WIDTH = 256
    ORIG_HEIGHT = 896
    TARGET_WIDTH = 192
    TARGET_HEIGHT = 640

    def __init__(self, excel_file, image_dir, transform=None, augment=False):
        self.data = pd.read_excel(excel_file)
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment
        # tiny translate/scale only
        self.aug = transforms.RandomApply([
            transforms.RandomAffine(degrees=0,
                                     translate=(0.01,0.01),
                                     scale=(0.98,1.02))
        ], p=0.5)
        self.grouped = self.data.groupby(['PatientID','Sample'])
        self.sample_keys = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        pid, sample = self.sample_keys[idx]
        group = self.grouped.get_group((pid, sample))
        img_path = os.path.join(self.image_dir, f"sample{sample}-{pid}-resized.jpg")
        img = Image.open(img_path).convert('L')

        if self.augment and self.transform:
            img = self.aug(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT))
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        coords = torch.zeros(12, dtype=torch.float32)
        label_map = {'RH':0,'RK':1,'RA':2,'LH':3,'LK':4,'LA':5}
        for _, row in group.iterrows():
            i = label_map[row['Label']]
            coords[2*i]   = row['X'] / self.ORIG_WIDTH
            coords[2*i+1] = row['Y'] / self.ORIG_HEIGHT
        return img, coords