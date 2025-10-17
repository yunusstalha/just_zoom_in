import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

from configs.base import get_config
from utils.visualization_utils import load_full_satellite_path_with_details, draw_grid_and_marker, tensor_to_image
from data.dataset import ZoomDataset
from data.transforms import build_transforms

# --- Configuration ---
NUM_SAMPLES_TO_VISUALIZE = 10
OUTPUT_DIRECTORY = "./dataset_visualizations/sample_visualizations"

def visualize_single_sample(cfg, sample, output_path):
    """
    Generates and saves a detailed visualization for a single data sample.
    """
    gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])
    sequence = sample["sequence"].numpy()
    
    # Use the centralized function to get all satellite images and their metadata
    path_details, final_patch = load_full_satellite_path_with_details(cfg, sequence, gt_latlon)

    # Ground images can have 2 crops. Handle this.
    ground_images = tensor_to_image(sample["ground"], denorm=True)
    if ground_images.ndim == 3: # If only one crop, add a batch dimension
        ground_images = np.expand_dims(ground_images, axis=0)
    num_ground_crops = ground_images.shape[0]

    # Setup plot layout
    total_cols = num_ground_crops + cfg.data.sequence_length + 1
    width_ratios = [1.5] * num_ground_crops + [1] * (cfg.data.sequence_length + 1)
    
    fig, axes = plt.subplots(1, total_cols, figsize=(3.5 * total_cols, 4), gridspec_kw={'width_ratios': width_ratios})
    
    # Plot Ground query images
    for i in range(num_ground_crops):
        axes[i].imshow(ground_images[i])
        axes[i].set_title(f"Query Crop {i+1}")
        axes[i].axis('off')

    # Plot each step in the zoom sequence
    for j in range(cfg.data.sequence_length):
        ax = axes[num_ground_crops + j]
        details = path_details[j]
        ax.imshow(details["image_raw"])
        ax.set_title(f"Step {j+1}\nPatch: {details['patch_index']}")
        draw_grid_and_marker(
            ax=ax, image_shape=details["image_raw"].shape, grid_size=cfg.data.grid_size,
            gt_pixel_coords=details["gt_pixels"],
            selected_patch_coords=divmod(details['patch_index'], cfg.data.grid_size),
        )

    # Plot the final, high-resolution view
    ax_final = axes[-1]
    ax_final.imshow(final_patch["image_raw"])
    ax_final.set_title(f"Final View (Lvl {cfg.data.sequence_length})")
    ax_final.plot(final_patch["gt_pixels"][0], final_patch["gt_pixels"][1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=1.0)
    ax_final.axis('off')

    fig.suptitle(f"Ground Truth Sequence for ID: {sample['meta']['image_id']}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # We need the dataset to get samples
    transforms = build_transforms(cfg.data.target_image_size)
    dataset = ZoomDataset(cfg, transforms=transforms)
    
    num_to_visualize = min(NUM_SAMPLES_TO_VISUALIZE, len(dataset))
    indices = np.random.choice(len(dataset), num_to_visualize, replace=False)
    
    print(f"Generating {num_to_visualize} visualizations...")
    for idx in tqdm(indices, desc="Generating Visualizations"):
        sample = dataset[idx]
        output_path = os.path.join(OUTPUT_DIRECTORY, f"gt_sequence_{sample['meta']['image_id']}.png")
        visualize_single_sample(cfg, sample, output_path)
            
    print(f"\nDone. Saved {num_to_visualize} visualizations to '{OUTPUT_DIRECTORY}'")

if __name__ == "__main__":
    main()