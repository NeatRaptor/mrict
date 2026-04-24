import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import csv
from pytorch_msssim import ssim as ssim_fn

from scripts.dataset import get_loader
from scripts.models import Generator, Discriminator
from scripts.utils import save_sample, compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_GEN      = 2e-4
LR_DISC     = 1e-4
BATCH_SIZE  = 4
NUM_EPOCHS  = 200

LAMBDA_L1   = 100
LAMBDA_SSIM = 10

WARMUP_EPOCHS = 5

CHECKPOINT_DIR = "checkpoints"
METRICS_FILE   = "metrics_log.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

bce = nn.BCEWithLogitsLoss()
l1  = nn.L1Loss()
# ─────────────────────────────────────────────────────────────────────────────


def init_metrics_file():
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "loss_D",
                "loss_G_GAN",
                "loss_G_L1",
                "loss_G_SSIM",
                "val_SSIM",
                "val_PSNR_dB"
            ])
        print(f"Created metrics log: {METRICS_FILE}")


def log_metrics(epoch, loss_D, loss_G_GAN, loss_G_L1, loss_G_SSIM, val_ssim, val_psnr):
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{loss_D:.5f}",
            f"{loss_G_GAN:.5f}",
            f"{loss_G_L1:.5f}",
            f"{loss_G_SSIM:.5f}",
            f"{val_ssim:.5f}",
            f"{val_psnr:.3f}"
        ])


def save_checkpoint(epoch, gen, disc, opt_gen, opt_disc):
    checkpoint = {
        "epoch":    epoch,
        "gen":      gen.state_dict(),
        "disc":     disc.state_dict(),
        "opt_gen":  opt_gen.state_dict(),
        "opt_disc": opt_disc.state_dict(),
    }
    torch.save(checkpoint, f"{CHECKPOINT_DIR}/latest.pth")
    if epoch % 10 == 0:
        torch.save(checkpoint, f"{CHECKPOINT_DIR}/epoch_{epoch}.pth")
        print(f"  Saved backup checkpoint: epoch_{epoch}.pth")


def load_checkpoint(path, gen, disc, opt_gen, opt_disc):
    ckpt = torch.load(path, map_location=DEVICE)
    gen.load_state_dict(ckpt["gen"])
    disc.load_state_dict(ckpt["disc"])
    opt_gen.load_state_dict(ckpt["opt_gen"])
    opt_disc.load_state_dict(ckpt["opt_disc"])
    print(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
def main():
    gen  = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    opt_gen  = optim.Adam(gen.parameters(),  lr=LR_GEN,  betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

    sched_gen  = CosineAnnealingLR(opt_gen,  T_max=NUM_EPOCHS, eta_min=1e-6)
    sched_disc = CosineAnnealingLR(opt_disc, T_max=NUM_EPOCHS, eta_min=1e-6)

    loader = get_loader(BATCH_SIZE)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    init_metrics_file()

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    start_epoch = 0
    if os.path.exists(f"{CHECKPOINT_DIR}/latest.pth"):
        start_epoch = load_checkpoint(
            f"{CHECKPOINT_DIR}/latest.pth",
            gen, disc, opt_gen, opt_disc
        )

    print(f"Using device  : {DEVICE}")
    print(f"Dataset size  : {len(loader.dataset)} slices")
    print(f"Training from : epoch {start_epoch + 1} to {NUM_EPOCHS}")
    print(f"Warmup epochs : {WARMUP_EPOCHS} (discriminator disabled)")
    print(f"Metrics log   : {METRICS_FILE}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        gen.train()
        disc.train()

        in_warmup = epoch < WARMUP_EPOCHS
        lambda_l1 = 50 if in_warmup else LAMBDA_L1

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        running = {"D": 0.0, "G_GAN": 0.0, "G_L1": 0.0, "G_SSIM": 0.0}

        for batch_idx, (mri, ct) in enumerate(loop):
            mri, ct = mri.to(DEVICE), ct.to(DEVICE)

            # ── Train Discriminator (every 2 batches, skip during warmup) ─────
            if not in_warmup and batch_idx % 2 == 0:
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    fake_ct_d  = gen(mri)
                    real_pair  = torch.cat([mri, ct],               dim=1)
                    fake_pair  = torch.cat([mri, fake_ct_d.detach()], dim=1)
                    D_real     = disc(real_pair)
                    D_fake     = disc(fake_pair)
                    loss_D     = (
                        bce(D_real, torch.full_like(D_real, 0.9)) +
                        bce(D_fake, torch.zeros_like(D_fake))
                    ) / 2

                opt_disc.zero_grad(set_to_none=True)
                scaler.scale(loss_D).backward()
                scaler.step(opt_disc)
                running["D"] += loss_D.item()

            # ── Train Generator TWICE per batch ───────────────────────────────
            for _ in range(2):
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    fake_ct = gen(mri)

                    if not in_warmup:
                        fake_pair  = torch.cat([mri, fake_ct], dim=1)
                        D_fake     = disc(fake_pair)
                        loss_G_GAN = bce(D_fake, torch.ones_like(D_fake))
                    else:
                        loss_G_GAN = torch.tensor(0.0, device=DEVICE)

                    loss_G_L1   = l1(fake_ct, ct)
                    loss_G_SSIM = 1 - ssim_fn(
                        (fake_ct + 1) / 2,
                        (ct      + 1) / 2,
                        data_range=1.0, size_average=True
                    )
                    loss_G = (
                        loss_G_GAN
                        + lambda_l1   * loss_G_L1
                        + LAMBDA_SSIM * loss_G_SSIM
                    )

                opt_gen.zero_grad(set_to_none=True)
                scaler.scale(loss_G).backward()
                scaler.step(opt_gen)
                scaler.update()

            running["G_GAN"]  += loss_G_GAN.item()
            running["G_L1"]   += loss_G_L1.item()
            running["G_SSIM"] += loss_G_SSIM.item()

            loop.set_postfix(
                warmup=in_warmup,
                D=f"{running['D'] / (batch_idx + 1):.3f}",
                G=f"{loss_G_GAN.item():.3f}",
                L1=f"{loss_G_L1.item():.4f}",
            )

        sched_gen.step()
        sched_disc.step()

        n          = len(loader)
        avg_D      = running["D"]      / n
        avg_G_GAN  = running["G_GAN"]  / n
        avg_G_L1   = running["G_L1"]   / n
        avg_G_SSIM = running["G_SSIM"] / n

        val_ssim, val_psnr = compute_metrics(gen, loader, DEVICE)

        status = "[WARMUP] " if in_warmup else ""
        print(
            f"{status}Epoch {epoch+1:>3}/{NUM_EPOCHS} | "
            f"D={avg_D:.3f}  G_GAN={avg_G_GAN:.3f}  "
            f"L1={avg_G_L1:.4f}  SSIM_loss={avg_G_SSIM:.4f} | "
            f"Val SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}dB"
        )

        log_metrics(epoch + 1, avg_D, avg_G_GAN, avg_G_L1, avg_G_SSIM, val_ssim, val_psnr)
        save_checkpoint(epoch + 1, gen, disc, opt_gen, opt_disc)
        save_sample(epoch + 1, gen, loader, DEVICE)

    print("Training complete!")
    print(f"All metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()