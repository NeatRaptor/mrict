# debug_export.py
import os

NPY_DIR = "eval_results/inferred_ct/npy"
MRI_DIR = "processed/brain/val/mri"

print("=== Inferred CT files ===")
if os.path.exists(NPY_DIR):
    files = sorted(os.listdir(NPY_DIR))
    print(f"Total files: {len(files)}")
    print("First 5:", files[:5])
else:
    print(f"FOLDER DOES NOT EXIST: {NPY_DIR}")

print("\n=== Val MRI files ===")
if os.path.exists(MRI_DIR):
    files = sorted(os.listdir(MRI_DIR))
    print(f"Total files: {len(files)}")
    print("First 5:", files[:5])
else:
    print(f"FOLDER DOES NOT EXIST: {MRI_DIR}")