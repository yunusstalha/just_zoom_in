from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.base import get_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms
from utils.visualization_utils import tensor_to_image

NUM_SAMPLES = 4          # How many consecutive samples to render starting at index 0.
USE_TRANSFORMS = True    # Use data/transforms.py (and denormalize for visualization).
OUTPUT_DIR = Path("./dataset_visualizations/dataset_inspection")

def save_sample(sample_idx: int, sample: dict, denorm: bool) -> None:
    sequence_indices = sample["sequence"].tolist()
    
    # Ground images can have multiple crops, visualize all of them
    ground_images = tensor_to_image(sample["ground"], denorm=denorm)
    if ground_images.ndim == 3: # If not batched, add a batch dimension
        ground_images = np.expand_dims(ground_images, axis=0)
    num_ground_crops = ground_images.shape[0]

    sat_sequence = tensor_to_image(sample["satellite_sequence"], denorm=denorm)
    
    panels = num_ground_crops + len(sequence_indices)
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4))
    if panels == 1:
        axes = [axes]

    # Plot Ground images
    for i in range(num_ground_crops):
        axes[i].imshow(ground_images[i])
        axes[i].set_title(f"Ground Crop {i+1}")
        axes[i].axis("off")

    # Plot Satellite sequence
    for i, patch_idx in enumerate(sequence_indices):
        ax_idx = num_ground_crops + i
        axes[ax_idx].imshow(sat_sequence[i])
        axes[ax_idx].set_title(f"Zoom {i+1}\nPatch {patch_idx}")
        axes[ax_idx].axis("off")

    fig.suptitle(
        f"Sample {sample_idx} – {sample['meta']['image_id']} "
        f"(lat {sample['meta']['latitude']:.5f}, lon {sample['meta']['longitude']:.5f})",
        fontsize=12,
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dataset_sample_{sample_idx:03d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    # Note: inspect_dataset should probably always use transforms to see what the model sees.
    # If USE_TRANSFORMS is False, ground images might not be cropped correctly.
    transforms = build_transforms(cfg.data.target_image_size) if USE_TRANSFORMS else None
    dataset = ZoomDataset(cfg, transforms=transforms)

    count = min(NUM_SAMPLES, len(dataset))
    indices = list(range(count))

    if not indices:
        raise ValueError("No valid indices to visualize.")

    for idx in indices:
        sample = dataset[idx]
        save_sample(idx, sample, denorm=USE_TRANSFORMS)