import os
import matplotlib.pyplot as plt

def save_sample(mri, ct, fake_ct, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mri = mri[0].cpu().detach().numpy().squeeze()
    ct = ct[0].cpu().detach().numpy().squeeze()
    fake_ct = fake_ct[0].cpu().detach().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mri, cmap='gray')
    axes[0].set_title("MRI")
    axes[0].axis("off")

    axes[1].imshow(ct, cmap='gray')
    axes[1].set_title("Real CT")
    axes[1].axis("off")

    axes[2].imshow(fake_ct, cmap='gray')
    axes[2].set_title("Fake CT")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}.png"))
    plt.close()