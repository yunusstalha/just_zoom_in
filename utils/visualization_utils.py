import torch
import numpy as np
import tiledwebmaps as twm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Constants ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Image Tensor Helpers  ---

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalizes an image tensor with ImageNet stats."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor: torch.Tensor, denorm: bool = True) -> np.ndarray:
    """Converts a torch image tensor to a numpy array for visualization."""
    image = tensor.detach().cpu()
    if denorm:
        # Handle batched ground crops (N, C, H, W) vs single images (C, H, W)
        if image.dim() == 4:
            image = torch.stack([_denormalize(img) for img in image])
        else:
            image = _denormalize(image)
    image = image.clamp(0.0, 1.0)
    return image.permute(0, 2, 3, 1).numpy() if image.dim() == 4 else image.permute(1, 2, 0).numpy()


# --- Core Drawing Functions ---

def draw_grid_and_marker(ax, image_shape, grid_size, gt_pixel_coords, selected_patch_coords, is_correct=None, draw_marker=True):
    """
    Draws a grid, a highlighted selected patch, and an optional ground-truth marker on a matplotlib axis.
    """
    img_h, img_w = image_shape[:2]
    patch_h, patch_w = img_h / grid_size, img_w / grid_size
    
    # Draw grid lines
    for i in range(1, grid_size):
        ax.axhline(i * patch_h, color='white', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.axvline(i * patch_w, color='white', linestyle='--', linewidth=0.7, alpha=0.7)

    # Determine highlight color based on prediction correctness
    if is_correct is None:
        color = 'yellow'  # Default for ground-truth row
    else:
        color = 'green' if is_correct else 'red'

    # Highlight the selected patch with a semi-transparent overlay
    sel_row, sel_col = selected_patch_coords
    rect = patches.Rectangle(
        (sel_col * patch_w, sel_row * patch_h), patch_w, patch_h,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.4
    )
    ax.add_patch(rect)
    
    # Conditionally plot GT location marker
    if draw_marker:
        ax.plot(gt_pixel_coords[0], gt_pixel_coords[1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=1.0)
    
    ax.axis('off')

# --- Geographic Calculation Helpers ---

def _calculate_gt_pixel_coords(gt_latlon, center_latlon, image_size_meters, image_shape):
    """Calculates the (x, y) pixel coordinates of a GT lat/lon on an image."""
    img_h, img_w = image_shape[:2]
    half_size_m = image_size_meters / 2.0
    
    north_center = twm.geo.move_from_latlon(center_latlon, 0, half_size_m)
    south_center = twm.geo.move_from_latlon(center_latlon, 180, half_size_m)
    top_left_latlon = twm.geo.move_from_latlon(north_center, 270, half_size_m)
    bottom_right_latlon = twm.geo.move_from_latlon(south_center, 90, half_size_m)

    lon_span = bottom_right_latlon[1] - top_left_latlon[1]
    lat_span = top_left_latlon[0] - bottom_right_latlon[0]
    frac_x = (gt_latlon[1] - top_left_latlon[1]) / lon_span if lon_span != 0 else 0.5
    frac_y = (top_left_latlon[0] - gt_latlon[0]) / lat_span if lat_span != 0 else 0.5
    
    return (frac_x * img_w, frac_y * img_h)

def _update_zoom_state(current_center, current_size, patch_index, grid_size):
    """Calculates the next zoom center and size based on the chosen patch."""
    patch_size = current_size / grid_size
    row, col = divmod(patch_index, grid_size)
    
    offset_east = (col + 0.5) * patch_size - (current_size / 2.0)
    offset_north = -((row + 0.5) * patch_size - (current_size / 2.0))
    
    temp_center = twm.geo.move_from_latlon(current_center, 90, offset_east)
    next_center = twm.geo.move_from_latlon(temp_center, 0, offset_north)
    
    return next_center, patch_size

# --- Main Data Loading Function for Visualization ---

def load_full_satellite_path_with_details(cfg, sequence, gt_latlon):
    """
    Dynamically loads satellite patches for a sequence and gathers all data
    needed for a detailed visualization, including the final zoomed-in patch.
    """
    tile_loader = twm.WithDefault(twm.from_yaml(cfg.paths.tile_layout))
    
    min_east, max_east, min_north, max_north = cfg.data.region_bounds_meters
    center_east = (min_east + max_east) / 2.0
    center_north = (min_north + max_north) / 2.0
    temp_center = twm.geo.move_from_latlon(np.array(cfg.data.geographic_center_latlon), 90, center_east)
    initial_center = twm.geo.move_from_latlon(temp_center, 0, center_north)
    initial_size = max_east - min_east

    path_details = []
    current_center = np.array(initial_center)
    current_size = initial_size
    
    # Use the target image size from the config for consistency
    target_image_size = cfg.data.target_image_size
    
    for patch_index in sequence:
        mpp = current_size / target_image_size
        array = tile_loader.load(
            latlon=current_center, bearing=0.0, meters_per_pixel=mpp,
            shape=(target_image_size, target_image_size),
        )
        
        path_details.append({
            "image_raw": array,
            "gt_pixels": _calculate_gt_pixel_coords(gt_latlon, current_center, current_size, array.shape),
            "patch_index": patch_index,
        })
        
        current_center, current_size = _update_zoom_state(current_center, current_size, patch_index, cfg.data.grid_size)

    # After the loop, load the final, high-resolution patch
    final_mpp = current_size / target_image_size
    final_array = tile_loader.load(
        latlon=current_center, bearing=0.0, meters_per_pixel=final_mpp,
        shape=(target_image_size, target_image_size),
    )
    
    final_patch_details = {
        "image_raw": final_array,
        "gt_pixels": _calculate_gt_pixel_coords(gt_latlon, current_center, current_size, final_array.shape),
    }

    return path_details, final_patch_details


def calculate_final_patch_bbox(sequence, grid_size):
    """
    Calculates the bounding box of the final patch relative to the initial image canvas.
    The bounding box is returned as fractional coordinates (0.0 to 1.0).
    
    Returns:
        (x_min, y_min, width, height)
    """
    # Start with the full canvas [x, y, w, h]
    x_min, y_min, width, height = 0.0, 0.0, 1.0, 1.0

    for patch_index in sequence:
        row, col = divmod(patch_index, grid_size)
        patch_width = width / grid_size
        patch_height = height / grid_size

        # Update the top-left corner of our new, smaller canvas
        x_min += col * patch_width
        y_min += row * patch_height

        # The size of our canvas shrinks for the next iteration
        width = patch_width
        height = patch_height

    return (x_min, y_min, width, height)


def generate_overview_visualization(cfg, gt_sequence, pred_sequence, gt_latlon, image_id, output_path):
    """
    Generates a high-res overview of the initial satellite map, highlighting the
    final GT and predicted patches.
    """
    HIGH_RES_SHAPE = (2048, 2048)
    
    # --- Step 1: Load the initial high-resolution satellite map ---
    tile_loader = twm.WithDefault(twm.from_yaml(cfg.paths.tile_layout))
    
    min_east, max_east, _, _ = cfg.data.region_bounds_meters
    center_east = (min_east + max_east) / 2.0
    center_north = (cfg.data.region_bounds_meters[2] + cfg.data.region_bounds_meters[3]) / 2.0
    temp_center = twm.geo.move_from_latlon(np.array(cfg.data.geographic_center_latlon), 90, center_east)
    initial_center = twm.geo.move_from_latlon(temp_center, 0, center_north)
    initial_size = max_east - min_east

    mpp = initial_size / HIGH_RES_SHAPE[0]
    initial_map_hr = tile_loader.load(
        latlon=initial_center, bearing=0.0, meters_per_pixel=mpp,
        shape=HIGH_RES_SHAPE
    )

    # --- Step 2: Calculate locations of patches and GT point ---
    gt_pixel_coords = _calculate_gt_pixel_coords(gt_latlon, initial_center, initial_size, HIGH_RES_SHAPE)
    
    gt_bbox_frac = calculate_final_patch_bbox(gt_sequence, cfg.data.grid_size)
    pred_bbox_frac = calculate_final_patch_bbox(pred_sequence, cfg.data.grid_size)

    # Convert fractional bboxes to absolute pixel coordinates for plotting
    h, w = HIGH_RES_SHAPE
    gt_bbox_pixels = (gt_bbox_frac[0] * w, gt_bbox_frac[1] * h, gt_bbox_frac[2] * w, gt_bbox_frac[3] * h)
    pred_bbox_pixels = (pred_bbox_frac[0] * w, pred_bbox_frac[1] * h, pred_bbox_frac[2] * w, pred_bbox_frac[3] * h)

    # --- Step 3: Plot the visualization ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(initial_map_hr)

    is_correct = np.array_equal(gt_sequence, pred_sequence)

    if is_correct:
        # Prediction is correct, draw one green saturated box
        rect = patches.Rectangle(
            (gt_bbox_pixels[0], gt_bbox_pixels[1]), gt_bbox_pixels[2], gt_bbox_pixels[3],
            linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.5, label='Correct Prediction'
        )
        ax.add_patch(rect)
    else:
        # Prediction is wrong, draw two distinct boxes
        rect_gt = patches.Rectangle(
            (gt_bbox_pixels[0], gt_bbox_pixels[1]), gt_bbox_pixels[2], gt_bbox_pixels[3],
            linewidth=2, edgecolor='gold', facecolor='none', label='Ground Truth Patch'
        )
        rect_pred = patches.Rectangle(
            (pred_bbox_pixels[0], pred_bbox_pixels[1]), pred_bbox_pixels[2], pred_bbox_pixels[3],
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.4, label='Predicted Patch'
        )
        ax.add_patch(rect_gt)
        ax.add_patch(rect_pred)

    # Plot the precise GT location as a dot
    ax.plot(gt_pixel_coords[0], gt_pixel_coords[1], 'go', markersize=8, markeredgecolor='white', markeredgewidth=1.5, label='Ground Truth Location')

    ax.set_title(f"Final Localization Overview for ID: {image_id}")
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)