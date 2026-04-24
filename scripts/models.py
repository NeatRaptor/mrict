import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Generator — U-Net
# ─────────────────────────────────────────────────────────────────────────────

class DownBlock(nn.Module):
    """
    Encoder block: Conv(stride=2) -> BN -> LeakyReLU
    Halves spatial resolution each step.
    No BN on the first block per original Pix2Pix paper.
    """
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder block: Upsample(bilinear) -> Conv -> BN -> Dropout -> ReLU

    WHY bilinear + Conv instead of ConvTranspose2d:
    ConvTranspose2d inserts zeros between activations before convolving.
    When stride does not divide evenly into kernel size, some output pixels
    receive more contributions than others, producing a regular grid of
    brighter/darker spots — the 'holes' or checkerboard pattern you observed.
    Bilinear upsample has no such overlap issue and produces smooth outputs.
    """
    def __init__(self, in_c, out_c, use_dropout=False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Full-depth U-Net following the Pix2Pix architecture.

    Architecture:
      - 7 encoder blocks + bottleneck
      - 7 decoder blocks with skip connections from matching encoder levels
      - Dropout on first 3 decoder blocks for regularization
      - Bilinear upsample in all decoder blocks to prevent checkerboard artifacts
      - Tanh output activation to match [-1, 1] target range

    Input:  (B, 1, 256, 256)  — single-channel MRI slice
    Output: (B, 1, 256, 256)  — synthetic CT slice
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.d1 = DownBlock(1,   64,  use_bn=False)  # (B,  64, 128, 128)
        self.d2 = DownBlock(64,  128)                 # (B, 128,  64,  64)
        self.d3 = DownBlock(128, 256)                 # (B, 256,  32,  32)
        self.d4 = DownBlock(256, 512)                 # (B, 512,  16,  16)
        self.d5 = DownBlock(512, 512)                 # (B, 512,   8,   8)
        self.d6 = DownBlock(512, 512)                 # (B, 512,   4,   4)
        self.d7 = DownBlock(512, 512)                 # (B, 512,   2,   2)

        # Bottleneck — no BN, acts as compressed latent representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),  # (B, 512, 1, 1)
            nn.ReLU(inplace=True)
        )

        # Decoder — in_c is doubled at each level due to skip connection concat
        self.u1 = UpBlock(512,    512, use_dropout=True)   # (B, 512,   2,   2)
        self.u2 = UpBlock(512*2,  512, use_dropout=True)   # (B, 512,   4,   4)
        self.u3 = UpBlock(512*2,  512, use_dropout=True)   # (B, 512,   8,   8)
        self.u4 = UpBlock(512*2,  512)                      # (B, 512,  16,  16)
        self.u5 = UpBlock(512*2,  256)                      # (B, 256,  32,  32)
        self.u6 = UpBlock(256*2,  128)                      # (B, 128,  64,  64)
        self.u7 = UpBlock(128*2,  64)                       # (B,  64, 128, 128)

        # Final output layer — bilinear upsample here too
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64*2, 1, 3, stride=1, padding=1),    # (B,   1, 256, 256)
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder path
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        bn = self.bottleneck(d7)

        # Decoder path with skip connections
        u1 = self.u1(bn)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator — 70x70 PatchGAN
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    70x70 PatchGAN discriminator from the original Pix2Pix paper.

    Rather than classifying the whole image as real/fake with a single value,
    PatchGAN outputs a spatial map where each value judges a 70x70 patch.
    This encourages the generator to produce realistic local textures,
    which is critical for medical image sharpness.

    Input:  (B, 2, 256, 256) — concatenated MRI + CT (real or fake)
    Output: (B, 1,  30,  30) — patch-level real/fake scores
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # No BN on first layer per original paper
            nn.Conv2d(2, 64, 4, stride=2, padding=1),               # (B,  64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64,  128, 4, stride=2, padding=1, bias=False), # (B, 128,  64,  64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False), # (B, 256,  32,  32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # stride=1 from here — maintains spatial resolution for patch output
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False), # (B, 512,  31,  31)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1,   4, stride=1, padding=1),             # (B,   1,  30,  30)
        )

    def forward(self, x):
        return self.model(x)