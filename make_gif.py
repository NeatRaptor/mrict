"""
make_gif.py — Export all inferred CT slices for a case as an animated GIF.
"""

import os
import numpy as np
from PIL import Image

CASE_ID  = "1BA034"
NPY_DIR  = "eval_results/inferred_ct/npy"
OUT_GIF  = f"matlab_export/{CASE_ID}.gif"
FPS      = 10          # frames per second — increase for faster scroll

os.makedirs("matlab_export", exist_ok=True)

files = sorted([f for f in os.listdir(NPY_DIR) if f.startswith(CASE_ID)],
               key=lambda f: int(f.rstrip(".npy").split("_")[-1]))
if not files:
    raise FileNotFoundError(f"No .npy files found for case '{CASE_ID}' in {NPY_DIR}")

frames = []
for fname in files:
    arr = np.load(os.path.join(NPY_DIR, fname))   # range [-1, 1]
    arr = (arr + 1) / 2                            # shift to [0, 1]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    frames.append(Image.fromarray(arr).convert("P"))  # palette mode for GIF

duration_ms = int(1000 / FPS)
frames = frames[::-1]
frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    loop=0,               # 0 = loop forever
    duration=duration_ms
)

print(f"Saved {len(frames)}-frame GIF -> {OUT_GIF}  ({FPS} fps)")