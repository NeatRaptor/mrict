"""
evaluate.py — Evaluation for the BRAIN Pix2Pix model.

Two modes:
  1. INFERENCE on val set (Task1_val) — MRI in, synthetic CT out, no metrics
     python evaluate.py --mode infer

  2. METRICS on train holdout — compares fake CT vs real CT, produces SSIM/PSNR/MAE
     python evaluate.py --mode metrics

Usage:
    python evaluate.py --mode infer   --checkpoint latest
    python evaluate.py --mode metrics --checkpoint epoch_80

Outputs (infer mode):
    eval_results/inferred_ct/   — synthetic CT .npy files + PNG visuals

Outputs (metrics mode):
    eval_results/metrics_per_slice.csv
    eval_results/metrics_summary.txt
    eval_results/samples/
"""

import torch
import numpy as np
import os
import csv
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from pytorch_msssim import ssim as ssim_fn

from scripts.models import Generator
from scripts.dataset import MRCTDataset

# ─────────────────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR       = "processed/brain/train"
VAL_MRI_DIR     = "processed/brain/val/mri"
CHECKPOINT_DIR  = "checkpoints"
OUTPUT_DIR      = "eval_results"
# ─────────────────────────────────────────────────────────────────────────────


class MRIOnlyDataset(Dataset):
    """Dataset for val split — loads MRI slices only, no CT."""
    def __init__(self, mri_dir):
        self.mri_dir = mri_dir
        self.files   = sorted(os.listdir(mri_dir))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {mri_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        mri   = np.load(os.path.join(self.mri_dir, fname)).astype(np.float32)
        mri   = np.clip(mri, -1.0, 1.0)
        mri   = torch.from_numpy(mri).unsqueeze(0)
        return mri, fname   # return filename so we can save outputs with matching names


def load_generator(checkpoint_path):
    gen  = Generator().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    gen.load_state_dict(ckpt["gen"])
    gen.eval()
    print(f"Loaded generator from : {checkpoint_path}")
    print(f"Checkpoint epoch      : {ckpt['epoch']}")
    return gen, ckpt["epoch"]


def compute_slice_metrics(fake, real):
    fake_01  = (fake + 1) / 2
    real_01  = (real + 1) / 2
    ssim_val = ssim_fn(fake_01, real_01, data_range=1.0, size_average=True).item()
    mse      = torch.mean((fake_01 - real_01) ** 2).item()
    psnr_val = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    mae_val  = torch.mean(torch.abs(fake_01 - real_01)).item()
    return ssim_val, psnr_val, mae_val


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(gen, epoch):
    """
    Run generator on val MRI slices and save synthetic CT outputs.
    No metrics computed since there is no ground truth CT.
    Outputs are saved as both .npy (for further use) and .png (for visual inspection).
    """
    if not os.path.exists(VAL_MRI_DIR):
        raise FileNotFoundError(
            f"Val MRI not found at {VAL_MRI_DIR}. "
            "Run python scripts/preprocess.py first."
        )

    infer_npy_dir = os.path.join(OUTPUT_DIR, "inferred_ct", "npy")
    infer_png_dir = os.path.join(OUTPUT_DIR, "inferred_ct", "png")
    os.makedirs(infer_npy_dir, exist_ok=True)
    os.makedirs(infer_png_dir, exist_ok=True)

    dataset = MRIOnlyDataset(VAL_MRI_DIR)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    print(f"\nRunning inference on {len(dataset)} val MRI slices...")

    for mri_batch, fnames in tqdm(loader, desc="Inferring"):
        mri_batch = mri_batch.to(DEVICE)
        fake_ct   = gen(mri_batch)

        for i in range(mri_batch.size(0)):
            fname    = fnames[i].replace(".npy", "")
            fake_arr = fake_ct[i, 0].cpu().numpy()

            # Save as .npy for potential downstream use
            np.save(os.path.join(infer_npy_dir, f"{fname}.npy"), fake_arr)

            # Save side-by-side PNG: MRI | Synthetic CT
            def to_01(x):
                return (x + 1) / 2

            grid = torch.cat([
                to_01(mri_batch[i:i+1]),
                to_01(fake_ct[i:i+1])
            ], dim=0)
            save_image(grid, os.path.join(infer_png_dir, f"{fname}.png"), nrow=2)

    print(f"\nInference complete.")
    print(f"  Synthetic CT .npy : {infer_npy_dir}/")
    print(f"  Visual PNGs       : {infer_png_dir}/")
    print(f"  Format            : [MRI input | Synthetic CT output]")


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_metrics(gen, epoch):
    """
    Compute SSIM, PSNR, MAE on training data (which has real CT for comparison).
    Uses the full training set as a proxy since val has no ground truth.
    """
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Train data not found at {TRAIN_DIR}")

    samples_dir    = os.path.join(OUTPUT_DIR, "samples")
    per_slice_path = os.path.join(OUTPUT_DIR, "metrics_per_slice.csv")
    summary_path   = os.path.join(OUTPUT_DIR, "metrics_summary.txt")
    os.makedirs(OUTPUT_DIR,   exist_ok=True)
    os.makedirs(samples_dir,  exist_ok=True)

    dataset = MRCTDataset(root_dir=TRAIN_DIR)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    print(f"\nComputing metrics on {len(dataset)} training slices...")

    all_ssim, all_psnr, all_mae = [], [], []

    with open(per_slice_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice_idx", "ssim", "psnr_dB", "mae"])

        for slice_idx, (mri, ct) in enumerate(tqdm(loader, desc="Computing metrics")):
            mri, ct = mri.to(DEVICE), ct.to(DEVICE)
            fake_ct = gen(mri)

            for i in range(mri.size(0)):
                s, p, m = compute_slice_metrics(fake_ct[i:i+1], ct[i:i+1])
                all_ssim.append(s)
                all_psnr.append(p)
                all_mae.append(m)
                writer.writerow([
                    slice_idx * loader.batch_size + i,
                    f"{s:.5f}", f"{p:.3f}", f"{m:.5f}"
                ])

            # Save visual samples every 50 batches
            if slice_idx % 50 == 0:
                def to_01(x):
                    return (x + 1) / 2
                grid = torch.cat([to_01(mri), to_01(fake_ct), to_01(ct)], dim=0)
                save_image(
                    grid,
                    os.path.join(samples_dir, f"sample_{slice_idx:04d}.png"),
                    nrow=mri.size(0)
                )

    ssim_arr = np.array(all_ssim)
    psnr_arr = np.array(all_psnr)
    mae_arr  = np.array(all_mae)

    summary_lines = [
        f"Pix2Pix Brain — Evaluation Results",
        f"====================================",
        f"Checkpoint epoch  : {epoch}",
        f"Slices evaluated  : {len(all_ssim)}",
        f"Device            : {DEVICE}",
        f"",
        f"Metric        Mean     Std      Min      Max",
        f"----------------------------------------------------",
        f"SSIM        {ssim_arr.mean():.4f}   {ssim_arr.std():.4f}   "
        f"{ssim_arr.min():.4f}   {ssim_arr.max():.4f}",
        f"PSNR (dB)   {psnr_arr.mean():.3f}   {psnr_arr.std():.3f}   "
        f"{psnr_arr.min():.3f}   {psnr_arr.max():.3f}",
        f"MAE         {mae_arr.mean():.4f}   {mae_arr.std():.4f}   "
        f"{mae_arr.min():.4f}   {mae_arr.max():.4f}",
        f"",
        f"Note: Metrics computed on training data (val has no ground truth CT).",
        f"",
        f"Outputs saved to  : {OUTPUT_DIR}/",
        f"  Per-slice CSV   : {per_slice_path}",
        f"  Sample images   : {samples_dir}/",
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\nSummary saved to: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="infer",
        choices=["infer", "metrics"],
        help="'infer' = run on val set and save synthetic CTs. "
             "'metrics' = compute SSIM/PSNR/MAE on training data."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="latest",
        help="Checkpoint name without .pth (e.g. 'latest' or 'epoch_80')"
    )
    args = parser.parse_args()

    checkpoint_path = f"{CHECKPOINT_DIR}/{args.checkpoint}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gen, epoch = load_generator(checkpoint_path)

    if args.mode == "infer":
        run_inference(gen, epoch)
    else:
        run_metrics(gen, epoch)


if __name__ == "__main__":
    main()