# export_to_mat.py
import numpy as np
import os
import scipy.io as sio

CASE_ID    = "1BA004"   # change to whichever case you want to visualise
NPY_DIR    = "eval_results/inferred_ct/npy"
OUTPUT_MAT = "matlab_export/synthetic_ct.mat"

os.makedirs("matlab_export", exist_ok=True)

ct_files = sorted([f for f in os.listdir(NPY_DIR) if f.startswith(CASE_ID)],
               key=lambda f: int(f.rstrip('.npy').split('_')[-1]))

if not ct_files:
    raise FileNotFoundError(f"No .npy files found for case '{CASE_ID}' in {NPY_DIR}")

ct_vol = np.stack([np.load(os.path.join(NPY_DIR, f)) for f in ct_files], axis=2)

# Shift from [-1, 1] to [0, 1] for MATLAB
ct_vol = (ct_vol + 1) / 2

sio.savemat(OUTPUT_MAT, {
    "synthetic_ct": ct_vol,
    "case_id":      CASE_ID
})

print(f"Saved volume: {ct_vol.shape} -> {OUTPUT_MAT}")