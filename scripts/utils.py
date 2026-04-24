import torch
import os
from torchvision.utils import save_image
from pytorch_msssim import ssim as ssim_fn


def save_sample(epoch, gen, loader, device, output_dir="outputs"):
    """
    Save a side-by-side grid of [MRI | Fake CT | Real CT] for visual inspection.
    One row per sample in the batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    gen.eval()

    mri, ct = next(iter(loader))
    mri, ct = mri.to(device), ct.to(device)

    with torch.no_grad():
        fake_ct = gen(mri)

    # Shift [-1, 1] -> [0, 1] for saving
    def to_01(x):
        return (x + 1) / 2

    grid = torch.cat([to_01(mri), to_01(fake_ct), to_01(ct)], dim=0)
    save_image(grid, f"{output_dir}/epoch_{epoch}.png", nrow=mri.size(0))
    gen.train()


@torch.no_grad()
def compute_metrics(gen, loader, device, num_batches=5):
    """
    Compute average SSIM and PSNR over a few validation batches.

    SSIM (Structural Similarity Index):
      - Range: 0 to 1. Higher is better.
      - Target for brain MRI->CT: > 0.85

    PSNR (Peak Signal-to-Noise Ratio):
      - Unit: dB. Higher is better.
      - Target for brain MRI->CT: > 28 dB

    These are the two standard metrics reported in SynthRAD2023
    and most medical image synthesis papers.
    """
    gen.eval()

    ssim_total = 0.0
    psnr_total = 0.0
    count      = 0

    for i, (mri, ct) in enumerate(loader):
        if i >= num_batches:
            break

        mri, ct = mri.to(device), ct.to(device)
        fake_ct = gen(mri)

        # Shift to [0, 1] for metric computation
        fake_01 = (fake_ct + 1) / 2
        real_01 = (ct      + 1) / 2

        # SSIM
        s = ssim_fn(fake_01, real_01, data_range=1.0, size_average=True)
        ssim_total += s.item()

        # PSNR
        mse = torch.mean((fake_01 - real_01) ** 2)
        if mse.item() > 1e-10:
            psnr = 10 * torch.log10(torch.tensor(1.0) / mse)
            psnr_total += psnr.item()

        count += 1

    gen.train()

    if count == 0:
        return 0.0, 0.0

    return ssim_total / count, psnr_total / count