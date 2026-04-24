"""
preprocess.py — Preprocessing pipeline for BRAIN data (SynthRAD2023 Task1).

Train split: requires mr.nii.gz + ct.nii.gz + mask.nii.gz
Val split  : requires mr.nii.gz + mask.nii.gz only (no CT in Task1_val)
"""

import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_ROOT = "Task1/brain"
TRAIN_SAVE = "processed/brain/train"

VAL_ROOT   = "Task1_val/brain"
VAL_SAVE   = "processed/brain/val"

IMG_SIZE         = 256
SLICE_THRESHOLD  = 15000
TISSUE_FRACTION  = 0.15
ZSCORE_CLIP      = 3.0
# ─────────────────────────────────────────────────────────────────────────────


def reorient_to_ras(img):
    return nib.as_closest_canonical(img)


def normalize_mri(mri, mask):
    brain_voxels = mri[mask > 0]
    mean = brain_voxels.mean()
    std  = brain_voxels.std() + 1e-8
    mri  = (mri - mean) / std
    mri  = np.clip(mri, -ZSCORE_CLIP, ZSCORE_CLIP)
    mri  = mri / ZSCORE_CLIP
    return mri.astype(np.float32)


def normalize_ct(ct, mask):
    ct_norm = np.clip(ct, -1000, 2000)
    ct_norm = (ct_norm + 1000) / 3000
    ct_norm = 2 * ct_norm - 1
    ct_norm = ct_norm * mask + (-1.0) * (1.0 - mask)
    return ct_norm.astype(np.float32)


def process_case_train(case_path, save_mri_dir, save_ct_dir, case_id):
    """
    Training case — needs MRI, CT, and mask.
    Saves paired mri/ct .npy files.
    """
    mri_path  = os.path.join(case_path, "mr.nii.gz")
    ct_path   = os.path.join(case_path, "ct.nii.gz")
    mask_path = os.path.join(case_path, "mask.nii.gz")

    for p in [mri_path, ct_path, mask_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] {case_id} — missing {os.path.basename(p)}")
            return 0

    mri_img  = reorient_to_ras(nib.load(mri_path))
    ct_img   = reorient_to_ras(nib.load(ct_path))
    mask_img = reorient_to_ras(nib.load(mask_path))

    mri  = mri_img.get_fdata().astype(np.float32)
    ct   = ct_img.get_fdata().astype(np.float32)
    mask = mask_img.get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)

    mri = normalize_mri(mri, mask)
    ct  = normalize_ct(ct, mask)
    mri = mri * mask + (-1.0) * (1.0 - mask)

    slice_count = 0

    for i in range(mri.shape[2]):
        mri_slice  = mri[:, :, i]
        ct_slice   = ct[:, :, i]
        mask_slice = mask[:, :, i]

        if np.sum(mask_slice) < SLICE_THRESHOLD:
            continue
        if ct_slice.max() - ct_slice.min() < 0.5:
            continue
        if np.mean(ct_slice > -0.8) < TISSUE_FRACTION:
            continue

        mri_slice = cv2.resize(mri_slice, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)
        ct_slice  = cv2.resize(ct_slice,  (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        fname = f"{case_id}_{slice_count}.npy"
        np.save(os.path.join(save_mri_dir, fname), mri_slice)
        np.save(os.path.join(save_ct_dir,  fname), ct_slice)
        slice_count += 1

    return slice_count


def process_case_val(case_path, save_mri_dir, case_id):
    """
    Validation case — MRI and mask only, no CT available.
    Saves MRI .npy files only. CT is what the model will predict.
    """
    mri_path  = os.path.join(case_path, "mr.nii.gz")
    mask_path = os.path.join(case_path, "mask.nii.gz")

    for p in [mri_path, mask_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] {case_id} — missing {os.path.basename(p)}")
            return 0

    mri_img  = reorient_to_ras(nib.load(mri_path))
    mask_img = reorient_to_ras(nib.load(mask_path))

    mri  = mri_img.get_fdata().astype(np.float32)
    mask = mask_img.get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)

    mri = normalize_mri(mri, mask)
    mri = mri * mask + (-1.0) * (1.0 - mask)

    slice_count = 0

    for i in range(mri.shape[2]):
        mri_slice  = mri[:, :, i]
        mask_slice = mask[:, :, i]

        # Filter using mask only since we have no CT
        if np.sum(mask_slice) < SLICE_THRESHOLD:
            continue
        if np.mean(mri_slice > -0.8) < TISSUE_FRACTION:
            continue

        mri_slice = cv2.resize(mri_slice, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        fname = f"{case_id}_{slice_count}.npy"
        np.save(os.path.join(save_mri_dir, fname), mri_slice)
        slice_count += 1

    return slice_count


def process_train_split():
    mri_save = os.path.join(TRAIN_SAVE, "mri")
    ct_save  = os.path.join(TRAIN_SAVE, "ct")
    os.makedirs(mri_save, exist_ok=True)
    os.makedirs(ct_save,  exist_ok=True)

    cases = sorted([
        c for c in os.listdir(TRAIN_ROOT)
        if os.path.isdir(os.path.join(TRAIN_ROOT, c))
    ])

    print(f"\nProcessing TRAIN: {len(cases)} cases from {TRAIN_ROOT}")
    total = 0
    for case in tqdm(cases):
        total += process_case_train(
            os.path.join(TRAIN_ROOT, case),
            mri_save, ct_save, case
        )
    print(f"TRAIN done — {total} slices saved to {TRAIN_SAVE}")


def process_val_split():
    if not os.path.exists(VAL_ROOT):
        print(f"\n[SKIP] VAL — folder not found: {VAL_ROOT}")
        return

    # Val only needs mri folder — no ct folder since we have no ground truth
    mri_save = os.path.join(VAL_SAVE, "mri")
    os.makedirs(mri_save, exist_ok=True)

    cases = sorted([
        c for c in os.listdir(VAL_ROOT)
        if os.path.isdir(os.path.join(VAL_ROOT, c))
    ])

    print(f"\nProcessing VAL: {len(cases)} cases from {VAL_ROOT}")
    print("Note: No CT available for val split — saving MRI only for inference.")
    total = 0
    for case in tqdm(cases):
        total += process_case_val(
            os.path.join(VAL_ROOT, case),
            mri_save, case
        )
    print(f"VAL done — {total} MRI slices saved to {VAL_SAVE}")


def main():
    process_train_split()
    process_val_split()


if __name__ == "__main__":
    main()