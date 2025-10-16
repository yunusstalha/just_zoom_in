import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Local application imports
from configs.base import get_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging

# Configure logging
setup_default_logging()
logger = logging.getLogger(__name__)

def main():
    # 1. --- Configuration and Initialization ---
    cfg = get_config()
    
    # Manually set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # 2. --- Data Loading ---
    transforms = build_transforms(cfg.data.target_image_size)
    dataset = ZoomDataset(cfg, transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Successfully loaded dataset with {len(dataset)} samples.")

    # 3. --- Model, Optimizer, and Loss ---
    model = GeoLocalizationModel(cfg)
    model.to(device) # Move model to the selected device
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 4. --- Training Loop ---
    total_steps = 0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Manually move data to the correct device
            ground_images = batch["ground"].to(device)
            satellite_sequence = batch["satellite_sequence"].to(device)
            target_sequence = batch["sequence"].to(device)

            # --- Forward pass ---
            logits = model(
                ground_images=ground_images,
                satellite_sequence=satellite_sequence,
                target_sequence=target_sequence
            )

            # --- Loss calculation ---
            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                target_sequence.view(-1)
            )

            # --- Backward pass and optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            
            # --- Logging ---
            if step % cfg.training.log_interval == 0:
                loss_val = loss.item()
                # Log to console
                # logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss_val:.4f}")
                # Update tqdm progress bar
                progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
        logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss_val:.4f}")

        # --- Save a checkpoint at the end of each epoch ---
        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training finished.")

if __name__ == "__main__":
    main()



# import logging
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from pathlib import Path
# import matplotlib.pyplot as plt
# from PIL import Image
# import tiledwebmaps as twm
# import numpy as np

# # Local application imports
# from configs.base import get_config
# from data.dataset import ZoomDataset
# from data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
# from models.model import GeoLocalizationModel
# from utils.logger import setup_default_logging

# # Configure logging
# setup_default_logging()
# logger = logging.getLogger(__name__)

# # --- Visualization Helper Functions (adapted from utils) ---

# def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
#     """Denormalizes a tensor image with ImageNet stats."""
#     mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
#     std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
#     return tensor * std + mean

# def _tensor_to_image(tensor: torch.Tensor, denorm: bool = True) -> np.ndarray:
#     """Converts a torch tensor to a numpy image for plotting."""
#     image = tensor.detach().cpu()
#     if denorm:
#         image = _denormalize(image)
#     image = image.clamp(0.0, 1.0)
#     return image.permute(1, 2, 0).numpy()

# def load_predicted_satellite_path(cfg, sequence):
#     """Dynamically loads a sequence of satellite patches based on predicted actions."""
#     tile_loader = twm.WithDefault(twm.from_yaml(cfg.paths.tile_layout))
#     transforms = build_transforms(cfg.data.target_image_size)
#     transform_sat = transforms["satellite"]

#     # Calculate initial canvas state (from dataset.py)
#     min_east, max_east, min_north, max_north = cfg.data.region_bounds_meters
#     center_east = (min_east + max_east) / 2.0
#     center_north = (min_north + max_north) / 2.0
#     temp_center = twm.geo.move_from_latlon(np.array(cfg.data.geographic_center_latlon), 90, center_east)
#     initial_center = twm.geo.move_from_latlon(temp_center, 0, center_north)
#     initial_size = max_east - min_east

#     patches = []
#     current_center = np.array(initial_center)
#     current_size = initial_size

#     for patch_index in sequence:
#         meters_per_pixel = current_size / cfg.data.target_image_size
#         array = tile_loader.load(
#             latlon=current_center,
#             bearing=0.0,
#             meters_per_pixel=meters_per_pixel,
#             shape=(cfg.data.target_image_size, cfg.data.target_image_size),
#         )
#         image = Image.fromarray(array)
#         patches.append(transform_sat(image))

#         # Update center and size for the next zoom level
#         patch_size = current_size / cfg.data.grid_size
#         row, col = divmod(patch_index, cfg.data.grid_size)
#         offset_east = (col + 0.5) * patch_size - (current_size / 2.0)
#         offset_north = -((row + 0.5) * patch_size - (current_size / 2.0))
#         temp_center = twm.geo.move_from_latlon(current_center, 90, offset_east)
#         current_center = twm.geo.move_from_latlon(temp_center, 0, offset_north)
#         current_size = patch_size
        
#     return torch.stack(patches)


# @torch.no_grad()
# def run_inference_and_visualize(model, batch, cfg, device):
#     """Runs inference and generates visualizations for the overfitted batch."""
#     logger.info("--- Running Inference on Overfitted Batch ---")
#     model.eval()

#     # Get model predictions
#     logits = model(
#         ground_images=batch["ground"].to(device),
#         satellite_sequence=batch["satellite_sequence"].to(device),
#         target_sequence=batch["sequence"].to(device) # Still needed for teacher-forcing structure
#     )
#     predictions = torch.argmax(logits, dim=-1)

#     # Calculate accuracy
#     targets = batch["sequence"].to(device)
#     correct_predictions = (predictions == targets).sum()
#     total_predictions = targets.numel()
#     accuracy = (correct_predictions / total_predictions) * 100
    
#     logger.info(f"Overfit Batch Accuracy: {accuracy:.2f}%")
#     for i in range(cfg.data.sequence_length):
#         step_acc = (predictions[:, i] == targets[:, i]).float().mean() * 100
#         logger.info(f"  - Accuracy at Step {i+1}: {step_acc:.2f}%")

#     # --- Generate Visualization ---
#     num_samples_to_viz = min(4, cfg.training.batch_size)
#     output_dir = Path("./overfit_visualizations")
#     output_dir.mkdir(exist_ok=True)

#     for i in range(num_samples_to_viz):
#         gt_sequence = targets[i].cpu().numpy()
#         pred_sequence = predictions[i].cpu().numpy()
        
#         # Load satellite images for the predicted path
#         predicted_sat_imgs = load_predicted_satellite_path(cfg, pred_sequence)

#         fig, axes = plt.subplots(2, 1 + cfg.data.sequence_length, figsize=(12, 6))
        
#         # Plot Ground Truth row
#         axes[0, 0].imshow(_tensor_to_image(batch["ground"][i]))
#         axes[0, 0].set_title("Query")
#         axes[0, 0].set_ylabel("Ground Truth", fontweight='bold')
#         for j in range(cfg.data.sequence_length):
#             axes[0, j+1].imshow(_tensor_to_image(batch["satellite_sequence"][i][j]))
#             axes[0, j+1].set_title(f"Step {j+1}\nPatch: {gt_sequence[j]}")
            
#         # Plot Predicted row
#         axes[1, 0].imshow(_tensor_to_image(batch["ground"][i]))
#         axes[1, 0].set_title("Query")
#         axes[1, 0].set_ylabel("Predicted", fontweight='bold')
#         for j in range(cfg.data.sequence_length):
#             is_correct = (gt_sequence[j] == pred_sequence[j])
#             border_color = 'green' if is_correct else 'red'
#             axes[1, j+1].imshow(_tensor_to_image(predicted_sat_imgs[j]))
#             axes[1, j+1].set_title(f"Step {j+1}\nPred: {pred_sequence[j]}")
#             for spine in axes[1, j+1].spines.values():
#                 spine.set_edgecolor(border_color)
#                 spine.set_linewidth(3)

#         # Clean up axes
#         for ax in axes.flatten():
#             ax.set_xticks([])
#             ax.set_yticks([])

#         fig.suptitle(f"Sample {i} - GT: {gt_sequence} vs Pred: {pred_sequence}", fontsize=14)
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
        
#         save_path = output_dir / f"overfit_sample_{i}.png"
#         fig.savefig(save_path, dpi=150)
#         plt.close(fig)
#         logger.info(f"Saved visualization to {save_path}")

# def main():
#     # 1. --- Configuration and Initialization ---
#     cfg = get_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")
#     torch.manual_seed(42)

#     # 2. --- Data Loading ---
#     transforms = build_transforms(cfg.data.target_image_size)
#     dataset = ZoomDataset(cfg, transforms=transforms)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=cfg.training.batch_size,
#         shuffle=False, # Set to False to get a consistent batch for testing
#         num_workers=4,
#         pin_memory=True
#     )

#     logger.info("Fetching a single batch to overfit...")
#     single_batch = next(iter(dataloader))

#     # 3. --- Model, Optimizer, and Loss ---
#     model = GeoLocalizationModel(cfg)
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
#     criterion = torch.nn.CrossEntropyLoss()
    
#     # 4. --- Overfitting Loop ---
#     num_overfit_steps = 300 # Set the number of iterations
#     model.train()
    
#     progress_bar = tqdm(range(num_overfit_steps), desc="Overfitting on single batch")
    
#     for step in progress_bar:
#         # Move the single batch to the device in each iteration
#         ground_images = single_batch["ground"].to(device)
#         satellite_sequence = single_batch["satellite_sequence"].to(device)
#         target_sequence = single_batch["sequence"].to(device)

#         logits = model(
#             ground_images=ground_images,
#             satellite_sequence=satellite_sequence,
#             target_sequence=target_sequence
#         )
#         loss = criterion(
#             logits.view(-1, logits.shape[-1]),
#             target_sequence.view(-1)
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if step % 10 == 0:
#             progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

#     logger.info(f"Overfitting test finished. Final loss: {loss.item():.6f}")

#     # 5. --- Save Checkpoint ---
#     checkpoint_path = "./ckpt_overfitted.pth"
#     torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
#     logger.info(f"Saved overfitted model to {checkpoint_path}")

#     # 6. --- Run Inference and Visualization ---
#     run_inference_and_visualize(model, single_batch, cfg, device)

# if __name__ == "__main__":
#     main()