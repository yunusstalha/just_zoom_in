import torch
import numpy as np
import tiledwebmaps as twm
import matplotlib.patches as patches
from PIL import Image


# --- Core Drawing Functions ---

def draw_grid_and_marker(ax, image_shape, grid_size, gt_pixel_coords, selected_patch_coords, is_correct=None, draw_marker=True):
    """
    Draws a grid, a highlighted selected patch, and an optional ground-truth marker on a matplotlib axis.

    Args:
        ax: The matplotlib axis to draw on.
        image_shape: The (height, width) of the image.
        grid_size: The number of grid cells along one dimension (e.g., 4 for a 4x4 grid).
        gt_pixel_coords: A tuple (x, y) for the ground-truth location marker.
        selected_patch_coords: A tuple (row, col) of the selected patch to highlight.
        is_correct (bool, optional): If provided, colors the patch green for correct,
                                     red for incorrect. Defaults to yellow.
        draw_marker (bool): If True, draws the red dot for the GT location.
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
    
    for patch_index in sequence:
        mpp = current_size / cfg.data.target_image_size
        array = tile_loader.load(
            latlon=current_center, bearing=0.0, meters_per_pixel=mpp,
            shape=(cfg.data.target_image_size, cfg.data.target_image_size),
        )
        
        path_details.append({
            "image_raw": array,
            "gt_pixels": _calculate_gt_pixel_coords(gt_latlon, current_center, current_size, array.shape),
            "patch_index": patch_index,
        })
        
        current_center, current_size = _update_zoom_state(current_center, current_size, patch_index, cfg.data.grid_size)

    # After the loop, load the final, high-resolution patch
    final_mpp = current_size / cfg.data.target_image_size
    final_array = tile_loader.load(
        latlon=current_center, bearing=0.0, meters_per_pixel=final_mpp,
        shape=(cfg.data.target_image_size, cfg.data.target_image_size),
    )
    
    final_patch_details = {
        "image_raw": final_array,
        "gt_pixels": _calculate_gt_pixel_coords(gt_latlon, current_center, current_size, final_array.shape),
    }

    return path_details, final_patch_details