import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import tiledwebmaps as twm
import numpy as np

from configs.eval import get_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging
from utils.visualization_utils import (
    draw_grid_and_marker, 
    load_full_satellite_path_with_details,
    generate_overview_visualization,
    tensor_to_image
)

# --- Configuration ---
CHECKPOINT_PATH = "./checkpoints/epoch_50.pth"  
NUM_SAMPLES_TO_VIZ = 10
BATCH_SIZE = 64
OUTPUT_DIR = Path("./dataset_visualizations/evaluation_results")

GENERATE_DETAILED_VIZ = False 
GENERATE_OVERVIEW_VIZ = True 
# ---------------------

# Configure logging
setup_default_logging()
logger = logging.getLogger(__name__)


def generate_detailed_visualization(sample, gt_sequence, pred_sequence, cfg, output_path):

    gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])

    gt_path_details, gt_final_patch = load_full_satellite_path_with_details(cfg, gt_sequence, gt_latlon)
    pred_path_details, pred_final_patch = load_full_satellite_path_with_details(cfg, pred_sequence, gt_latlon)
    
    ground_images = tensor_to_image(sample["ground"], denorm=True)
    if ground_images.ndim == 3:
        ground_images = np.expand_dims(ground_images, axis=0)
    num_ground_crops = ground_images.shape[0]

    total_cols = num_ground_crops + cfg.data.sequence_length + 1
    width_ratios = [1.5] * num_ground_crops + [1] * (cfg.data.sequence_length + 1)
    
    fig, axes = plt.subplots(2, total_cols, figsize=(3.5 * total_cols, 7), gridspec_kw={'width_ratios': width_ratios})

    # --- Plot Ground Truth Row ---
    axes[0, 0].set_ylabel("Ground Truth", fontweight='bold', size='large')
    for i in range(num_ground_crops):
        axes[0, i].imshow(ground_images[i])
        axes[0, i].set_title(f"Query Crop {i+1}")
        axes[0, i].axis('off')

    for j in range(cfg.data.sequence_length):
        ax = axes[0, num_ground_crops + j]
        details = gt_path_details[j]
        ax.imshow(details["image_raw"])
        ax.set_title(f"Step {j+1}\nGT Patch: {details['patch_index']}")
        draw_grid_and_marker(
            ax=ax, image_shape=details["image_raw"].shape, grid_size=cfg.data.grid_size,
            gt_pixel_coords=details["gt_pixels"],
            selected_patch_coords=divmod(details['patch_index'], cfg.data.grid_size),
        )

    ax_final_gt = axes[0, -1]
    ax_final_gt.imshow(gt_final_patch["image_raw"])
    ax_final_gt.set_title(f"Final View (Lvl {cfg.data.sequence_length})")
    ax_final_gt.plot(gt_final_patch["gt_pixels"][0], gt_final_patch["gt_pixels"][1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=1.0)
    ax_final_gt.axis('off')

    # --- Plot Predicted Row ---
    axes[1, 0].set_ylabel("Predicted", fontweight='bold', size='large')
    for i in range(num_ground_crops):
        axes[1, i].imshow(ground_images[i])
        axes[1, i].set_title(f"Query Crop {i+1}")
        axes[1, i].axis('off')

    for j in range(cfg.data.sequence_length):
        ax = axes[1, num_ground_crops + j]
        details = pred_path_details[j]
        is_correct = (gt_sequence[j] == pred_sequence[j])
        ax.imshow(details["image_raw"])
        ax.set_title(f"Step {j+1}\nPred Patch: {details['patch_index']}")
        draw_grid_and_marker(
            ax=ax, image_shape=details["image_raw"].shape, grid_size=cfg.data.grid_size,
            gt_pixel_coords=details["gt_pixels"],
            selected_patch_coords=divmod(details['patch_index'], cfg.data.grid_size),
            is_correct=is_correct,
            draw_marker=False
        )

    ax_final_pred = axes[1, -1]
    ax_final_pred.imshow(pred_final_patch["image_raw"])
    ax_final_pred.set_title(f"Final View (Lvl {cfg.data.sequence_length})")
    ax_final_pred.axis('off')

    fig.suptitle(f"ID: {sample['meta']['image_id']} | GT: {gt_sequence} vs Pred: {pred_sequence}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def main():
    # 1. --- Setup ---
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint from: {CHECKPOINT_PATH}")

    # 2. --- Load Model from Checkpoint ---
    model = GeoLocalizationModel(cfg)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    # 3. --- Data Loading ---
    transforms = build_transforms(cfg.data.target_image_size)
    dataset = ZoomDataset(cfg, transforms=transforms)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    logger.info(f"Loaded dataset with {len(dataset)} samples for evaluation.")

    # 4. --- Full Evaluation for Accuracy Metrics ---
    all_preds, all_targets = [], []
    for batch in tqdm(dataloader, desc="Calculating accuracy"):
        logits = model(
            ground_images=batch["ground"].to(device),
            satellite_sequence=batch["satellite_sequence"].to(device),
            target_sequence=batch["sequence"].to(device)
        )
        all_preds.append(torch.argmax(logits, dim=-1).cpu())
        all_targets.append(batch["sequence"])

    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # --- Print Metrics ---
    overall_accuracy = (predictions == targets).float().mean().item() * 100
    logger.info(f"\n--- Evaluation Metrics ---")
    logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    for i in range(cfg.data.sequence_length):
        step_accuracy = (predictions[:, i] == targets[:, i]).float().mean().item() * 100
        logger.info(f"   - Accuracy at Step {i+1}: {step_accuracy:.2f}%")
    logger.info(f"--------------------------\n")

    # 5. --- Generate Visualizations for Random Samples ---
    logger.info(f"Generating visualizations for {NUM_SAMPLES_TO_VIZ} random samples...")
    
    num_samples = len(dataset)
    indices = np.random.choice(num_samples, min(NUM_SAMPLES_TO_VIZ, num_samples), replace=False)

    for idx in tqdm(indices, desc="Creating visualizations"):
        sample = dataset[idx]
        prediction_sequence = predictions[idx].numpy()
        gt_sequence = sample["sequence"].numpy()
        
        if GENERATE_DETAILED_VIZ:
            output_path = OUTPUT_DIR / f"eval_detailed_{sample['meta']['image_id']}.png"
            generate_detailed_visualization(sample, gt_sequence, prediction_sequence, cfg, output_path)

        if GENERATE_OVERVIEW_VIZ:
            gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])
            overview_output_path = OUTPUT_DIR / f"eval_overview_{sample['meta']['image_id']}.png"
            generate_overview_visualization(
                cfg,
                gt_sequence,
                prediction_sequence,
                gt_latlon,
                sample['meta']['image_id'],
                overview_output_path
            )

    logger.info(f"Visualizations saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()