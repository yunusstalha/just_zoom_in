import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import tiledwebmaps as twm
import numpy as np

# Local application imports
from configs.eval import get_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging

# --- Configuration ---
CHECKPOINT_PATH = "./checkpoints/epoch_50.pth"  # IMPORTANT: Set this to your trained model checkpoint
NUM_SAMPLES_TO_VIZ = 10
BATCH_SIZE = 64
OUTPUT_DIR = Path("./evaluation_results")
# ---------------------

# Configure logging
setup_default_logging()
logger = logging.getLogger(__name__)

# --- Helper Functions (copied from your previous script for completeness) ---

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu()
    image = _denormalize(image)
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()

def load_predicted_satellite_path(cfg, sequence):
    """Dynamically loads a sequence of satellite patches based on a predicted sequence."""
    tile_loader = twm.WithDefault(twm.from_yaml(cfg.paths.tile_layout))
    transforms = build_transforms(cfg.data.target_image_size)
    transform_sat = transforms["satellite"]
    
    # Calculate initial canvas state (from dataset.py)
    min_east, max_east, min_north, max_north = cfg.data.region_bounds_meters
    center_east = (min_east + max_east) / 2.0; center_north = (min_north + max_north) / 2.0
    temp_center = twm.geo.move_from_latlon(np.array(cfg.data.geographic_center_latlon), 90, center_east)
    initial_center = twm.geo.move_from_latlon(temp_center, 0, center_north)
    initial_size = max_east - min_east

    patches = []; current_center = np.array(initial_center); current_size = initial_size

    for patch_index in sequence:
        meters_per_pixel = current_size / cfg.data.target_image_size
        array = tile_loader.load(
            latlon=current_center, bearing=0.0, meters_per_pixel=meters_per_pixel,
            shape=(cfg.data.target_image_size, cfg.data.target_image_size),
        )
        patches.append(transform_sat(Image.fromarray(array)))

        patch_size = current_size / cfg.data.grid_size
        row, col = divmod(patch_index, cfg.data.grid_size)
        offset_east = (col + 0.5) * patch_size - (current_size / 2.0)
        offset_north = -((row + 0.5) * patch_size - (current_size / 2.0))
        temp_center = twm.geo.move_from_latlon(current_center, 90, offset_east)
        current_center = twm.geo.move_from_latlon(temp_center, 0, offset_north)
        current_size = patch_size
        
    return torch.stack(patches)

def generate_visualization(sample, gt_sequence, pred_sequence, cfg, output_path):
    """Creates and saves a single visualization plot."""
    # Load satellite images for the predicted path
    predicted_sat_imgs = load_predicted_satellite_path(cfg, pred_sequence)

    # We now have two ground crops, let's visualize them
    num_ground_crops = sample["ground"].shape[0]
    total_cols = num_ground_crops + cfg.data.sequence_length
    
    fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8), gridspec_kw={'width_ratios': [1]*num_ground_crops + [1]*cfg.data.sequence_length})

    # --- Plot Ground Truth Row ---
    axes[0, 0].set_ylabel("Ground Truth", fontweight='bold', size='large')
    for i in range(num_ground_crops):
        axes[0, i].imshow(_tensor_to_image(sample["ground"][i]))
        axes[0, i].set_title(f"Query Crop {i+1}")
    for j in range(cfg.data.sequence_length):
        axes[0, num_ground_crops + j].imshow(_tensor_to_image(sample["satellite_sequence"][j]))
        axes[0, num_ground_crops + j].set_title(f"Step {j+1}\nPatch: {gt_sequence[j]}")

    # --- Plot Predicted Row ---
    axes[1, 0].set_ylabel("Predicted", fontweight='bold', size='large')
    for i in range(num_ground_crops):
        axes[1, i].imshow(_tensor_to_image(sample["ground"][i]))
        axes[1, i].set_title(f"Query Crop {i+1}")
    for j in range(cfg.data.sequence_length):
        is_correct = (gt_sequence[j] == pred_sequence[j])
        border_color = 'green' if is_correct else 'red'
        ax = axes[1, num_ground_crops + j]
        ax.imshow(_tensor_to_image(predicted_sat_imgs[j]))
        ax.set_title(f"Step {j+1}\nPred: {pred_sequence[j]}")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)

    # Clean up axes
    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"ID: {sample['meta']['image_id']} | GT: {gt_sequence} vs Pred: {pred_sequence}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        logger.info(f"  - Accuracy at Step {i+1}: {step_accuracy:.2f}%")
    logger.info(f"--------------------------\n")

    # 5. --- Generate Visualizations for Random Samples ---
    logger.info(f"Generating visualizations for {NUM_SAMPLES_TO_VIZ} random samples...")
    
    # Get random indices for visualization
    num_samples = len(dataset)
    indices = np.random.choice(num_samples, NUM_SAMPLES_TO_VIZ, replace=False)

    for idx in tqdm(indices, desc="Creating visualizations"):
        sample = dataset[idx]
        # Get the corresponding prediction we already calculated
        prediction_sequence = predictions[idx].numpy()
        gt_sequence = sample["sequence"].numpy()
        
        output_path = OUTPUT_DIR / f"eval_{sample['meta']['image_id']}.png"
        generate_visualization(sample, gt_sequence, prediction_sequence, cfg, output_path)

    logger.info(f"Visualizations saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
