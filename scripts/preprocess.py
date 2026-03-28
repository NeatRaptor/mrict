import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
ROOT_DIR = "Task1/brain"
SAVE_DIR = "processed/brain/train"

IMG_SIZE = 256
SLICE_THRESHOLD = 500  # filter empty slices

# ---------------------------------------- #

def normalize_mri(mri):
    mri = (mri - np.min(mri)) / (np.max(mri) - np.min(mri) + 1e-8)
    mri = 2 * mri - 1  # [-1, 1]
    return mri

def normalize_ct(ct):
    ct = np.clip(ct, -1000, 2000)
    ct = (ct + 1000) / 3000
    ct = 2 * ct - 1  # [-1, 1]
    return ct

def process_case(case_path, save_mri_dir, save_ct_dir, case_id):
    mri_path = os.path.join(case_path, "mr.nii.gz")
    ct_path  = os.path.join(case_path, "ct.nii.gz")
    mask_path = os.path.join(case_path, "mask.nii.gz")

    if not os.path.exists(mri_path):
        return

    # Load volumes
    mri = nib.load(mri_path).get_fdata()
    ct  = nib.load(ct_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    # Apply mask
    mri = mri * mask
    ct  = ct * mask

    # Normalize
    mri = normalize_mri(mri)
    ct  = normalize_ct(ct)

    slice_count = 0

    for i in range(mri.shape[2]):
        mri_slice = mri[:, :, i]
        ct_slice  = ct[:, :, i]
        mask_slice = mask[:, :, i]

        # Skip empty slices
        if np.sum(mask_slice) < SLICE_THRESHOLD:
            continue

        # Resize
        mri_slice = cv2.resize(mri_slice, (IMG_SIZE, IMG_SIZE))
        ct_slice  = cv2.resize(ct_slice, (IMG_SIZE, IMG_SIZE))

        # Save
        np.save(os.path.join(save_mri_dir, f"{case_id}_{slice_count}.npy"), mri_slice)
        np.save(os.path.join(save_ct_dir, f"{case_id}_{slice_count}.npy"), ct_slice)

        slice_count += 1


def main():
    os.makedirs(os.path.join(SAVE_DIR, "mri"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "ct"), exist_ok=True)

    cases = sorted(os.listdir(ROOT_DIR))

    for case in tqdm(cases):
        case_path = os.path.join(ROOT_DIR, case)

        if not os.path.isdir(case_path):
            continue

        process_case(
            case_path,
            os.path.join(SAVE_DIR, "mri"),
            os.path.join(SAVE_DIR, "ct"),
            case
        )


if __name__ == "__main__":
    main()