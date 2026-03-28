import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "processed/brain/train"

mri_dir = os.path.join(DATA_DIR, "mri")
ct_dir  = os.path.join(DATA_DIR, "ct")

files = sorted(os.listdir(mri_dir))

# Pick a random sample
idx = np.random.randint(0, len(files))
file_name = files[idx]

mri = np.load(os.path.join(mri_dir, file_name))
ct  = np.load(os.path.join(ct_dir, file_name))

print("File:", file_name)
print("MRI shape:", mri.shape)
print("CT shape:", ct.shape)

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(mri, cmap="gray")
plt.title("MRI")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(ct, cmap="gray")
plt.title("CT")
plt.colorbar()

plt.show()

print("MRI min/max:", mri.min(), mri.max())
print("CT min/max:", ct.min(), ct.max())

for i in range(10):
    f = files[i]
    mri = np.load(os.path.join(mri_dir, f))
    ct  = np.load(os.path.join(ct_dir, f))

    assert mri.shape == ct.shape, f"Shape mismatch in {f}"

print("All shapes match!")

bad_count = 0

for f in files:
    mri = np.load(os.path.join(mri_dir, f))

    if np.sum(np.abs(mri)) < 10:  # almost empty
        bad_count += 1

print("Empty slices:", bad_count)

plt.figure(figsize=(12, 8))

for i in range(6):
    f = files[np.random.randint(len(files))]

    mri = np.load(os.path.join(mri_dir, f))
    ct  = np.load(os.path.join(ct_dir, f))

    plt.subplot(6, 2, 2*i+1)
    plt.imshow(mri, cmap="gray")
    plt.axis("off")

    plt.subplot(6, 2, 2*i+2)
    plt.imshow(ct, cmap="gray")
    plt.axis("off")

plt.suptitle("Random MRI-CT pairs")
plt.show()