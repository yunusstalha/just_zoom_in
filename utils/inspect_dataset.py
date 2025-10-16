from pathlib import Path

import matplotlib.pyplot as plt
import torch

from configs.base import get_config
from data.dataset import ZoomDataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms

# --- Basic knobs so we can tweak behaviour without editing much below ---
NUM_SAMPLES = 4          # How many consecutive samples to render starting at index 0.
USE_TRANSFORMS = False    # Use data/transforms.py (and denormalize for visualization).
OUTPUT_DIR = Path("./dataset_visualizations")


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def _tensor_to_image(tensor: torch.Tensor, denorm: bool) -> torch.Tensor:
    image = tensor.detach()
    if denorm:
        image = _denormalize(image)
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).cpu()


def save_sample(sample_idx: int, sample: dict, denorm: bool) -> None:
    sequence_indices = sample["sequence"].tolist()
    patches = sample["satellite_sequence"]
    panels = 1 + len(sequence_indices)

    fig, axes = plt.subplots(1, panels, figsize=(3 * panels, 3))
    if panels == 1:
        axes = [axes]

    axes[0].imshow(_tensor_to_image(sample["ground"], denorm))
    axes[0].set_title(f"Ground\n{sample['meta']['image_id']}")
    axes[0].axis("off")

    for col, (patch_tensor, patch_idx) in enumerate(zip(patches, sequence_indices), start=1):
        axes[col].imshow(_tensor_to_image(patch_tensor, denorm))
        axes[col].set_title(f"Zoom {col}\nPatch {patch_idx}")
        axes[col].axis("off")

    fig.suptitle(
        f"Sample {sample_idx} – {sample['meta']['image_id']} "
        f"(lat {sample['meta']['latitude']:.5f}, lon {sample['meta']['longitude']:.5f})",
        fontsize=10,
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dataset_sample_{sample_idx:03d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize_zoom_dataset] saved {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    transforms = build_transforms(cfg.data.target_image_size) if USE_TRANSFORMS else None
    dataset = ZoomDataset(cfg, transforms=transforms)

    count = min(NUM_SAMPLES, len(dataset))
    indices = list(range(count))

    if not indices:
        raise ValueError("No valid indices to visualize.")

    for idx in indices:
        sample = dataset[idx]
        save_sample(idx, sample, denorm=USE_TRANSFORMS)
