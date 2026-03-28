import sys
import os

# Fix import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import MRCTDataset
from models import Generator, Discriminator
from utils import save_sample


def main():
    # -------------------------------
    # Device
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    # -------------------------------
    # Hyperparameters
    # -------------------------------
    LR = 2e-4
    BATCH_SIZE = 8
    EPOCHS = 30
    LAMBDA_L1 = 50   # 🔥 reduced to reduce blur

    # -------------------------------
    # Paths
    # -------------------------------
    DATA_DIR = os.path.join(BASE_DIR, "processed", "brain", "train")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------
    # Dataset
    # -------------------------------
    full_dataset = MRCTDataset(DATA_DIR)

    subset_size = min(2000, len(full_dataset))
    indices = list(range(subset_size))
    dataset = Subset(full_dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset size:", len(dataset))

    # -------------------------------
    # Models
    # -------------------------------
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # -------------------------------
    # Optimizers
    # -------------------------------
    opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

    # -------------------------------
    # Loss
    # -------------------------------
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(EPOCHS):
        epoch_loss_D = 0
        epoch_loss_G = 0
        count = 0

        for i, batch in enumerate(loader):
            mri = batch["mri"].to(DEVICE, non_blocking=True)
            ct  = batch["ct"].to(DEVICE, non_blocking=True)

            # -----------------------
            # Train Discriminator
            # -----------------------
            fake_ct = gen(mri)

            D_real = disc(mri, ct)
            D_fake = disc(mri, fake_ct.detach())

            loss_D_real = bce(D_real, torch.ones_like(D_real))
            loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2

            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()

            # -----------------------
            # Train Generator
            # -----------------------
            D_fake = disc(mri, fake_ct)

            loss_G_GAN = bce(D_fake, torch.ones_like(D_fake))
            loss_G_L1 = l1(fake_ct, ct) * LAMBDA_L1

            loss_G = loss_G_GAN + loss_G_L1

            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            count += 1

        # -------------------------------
        # Epoch summary
        # -------------------------------
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Avg Loss D: {epoch_loss_D/count:.4f} | "
              f"Avg Loss G: {epoch_loss_G/count:.4f}")

        # -------------------------------
        # Save FULL IMAGE (important)
        # -------------------------------
        if (epoch + 1) % 5 == 0:
            save_sample(mri, ct, fake_ct, epoch+1, OUTPUT_DIR)

    print("Training complete!")


if __name__ == "__main__":
    main()