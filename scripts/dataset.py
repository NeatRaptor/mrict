import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MRCTDataset(Dataset):
    def __init__(self, root_dir="processed/brain/train"):
        self.ct_dir  = os.path.join(root_dir, "ct")
        self.mri_dir = os.path.join(root_dir, "mri")

        # Use CT file list as reference — both dirs should be identical
        self.files = sorted(os.listdir(self.ct_dir))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npy files found in {self.ct_dir}. "
                "Run scripts/preprocess.py first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        ct  = np.load(os.path.join(self.ct_dir,  fname)).astype(np.float32)
        mri = np.load(os.path.join(self.mri_dir, fname)).astype(np.float32)

        # ── DO NOT renormalize ────────────────────────────────────────────────
        # preprocess.py already outputs values in [-1, 1].
        # Renormalizing here corrupts the range. We only clamp as a safety net.
        ct  = np.clip(ct,  -1.0, 1.0)
        mri = np.clip(mri, -1.0, 1.0)

        # Add channel dimension: (H, W) -> (1, H, W)
        ct  = torch.from_numpy(ct).unsqueeze(0)
        mri = torch.from_numpy(mri).unsqueeze(0)

        return mri, ct


def get_loader(batch_size=4, root_dir="processed/brain/train", num_workers=4):
    dataset = MRCTDataset(root_dir=root_dir)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader