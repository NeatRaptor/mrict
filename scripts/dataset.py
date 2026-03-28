import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRCTDataset(Dataset):
    def __init__(self, root_dir):
        self.mri_dir = os.path.join(root_dir, "mri")
        self.ct_dir  = os.path.join(root_dir, "ct")

        self.files = sorted(os.listdir(self.mri_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        mri = np.load(os.path.join(self.mri_dir, file_name))
        ct  = np.load(os.path.join(self.ct_dir, file_name))

        # Add channel dimension
        mri = np.expand_dims(mri, axis=0)
        ct  = np.expand_dims(ct, axis=0)

        return {
            "mri": torch.tensor(mri, dtype=torch.float32),
            "ct": torch.tensor(ct, dtype=torch.float32)
        }