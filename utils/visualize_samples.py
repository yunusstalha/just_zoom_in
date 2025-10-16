import os
import sys
import json
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tiledwebmaps as twm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm


PV_IMAGES_PATH = "/local/scratch2/cross_view_data/filtered_street_v2/images"
AERIAL_DATASET_PATH = "/local/scratch2/cross_view_data/sat"
SEQUENCES_CSV_PATH = "./ground_truth_sequences.csv"

GEOGRAPHIC_CENTER_LATLON = np.array([38.8936, -77.0116])
REGION_BOUNDS_METERS = [700.0, 2700.0, -1000.0, 1000.0]

ZOOM_LEVELS = 3
GRID_SIZE = 4
TARGET_IMAGE_HEIGHT = 512
OUTPUT_DIRECTORY = "./sample_visualizations"


def draw_grid_and_marker(ax, image_shape, grid_size, gt_pixel_coords, selected_patch_coords):
    img_h, img_w = image_shape[:2]
    patch_h, patch_w = img_h / grid_size, img_w / grid_size
    for i in range(1, grid_size):
        ax.axhline(i * patch_h, color='white', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.axvline(i * patch_w, color='white', linestyle='-', linewidth=0.5, alpha=0.6)
    sel_row, sel_col = selected_patch_coords
    rect = patches.Rectangle(
        (sel_col * patch_w, sel_row * patch_h), patch_w, patch_h,
        linewidth=1, edgecolor='yellow', facecolor='yellow', alpha=0.4
    )
    ax.add_patch(rect)
    ax.plot(gt_pixel_coords[0], gt_pixel_coords[1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
    ax.axis('off')


def visualize_single_id(image_id, gt_latlon, sequence, tileloader):
    min_east_m, max_east_m = REGION_BOUNDS_METERS[0], REGION_BOUNDS_METERS[1]
    min_north_m, max_north_m = REGION_BOUNDS_METERS[2], REGION_BOUNDS_METERS[3]
    center_east_m = (min_east_m + max_east_m) / 2.0
    center_north_m = (min_north_m + max_north_m) / 2.0
    
    temp_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 90, center_east_m)
    current_center_latlon = twm.geo.move_from_latlon(temp_center, 0, center_north_m)
    current_image_size_m = max_east_m - min_east_m
    
    visualization_data = []
    for k in range(ZOOM_LEVELS):
        patch_index = sequence[k]
        grid_row, grid_col = patch_index // GRID_SIZE, patch_index % GRID_SIZE

        mpp = current_image_size_m / TARGET_IMAGE_HEIGHT
        aerial_image = tileloader.load(
            latlon=current_center_latlon, bearing=0.0, meters_per_pixel=mpp,
            shape=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_HEIGHT)
        )

        half_size_m = current_image_size_m / 2.0
        north_center = twm.geo.move_from_latlon(current_center_latlon, 0, half_size_m)
        south_center = twm.geo.move_from_latlon(current_center_latlon, 180, half_size_m)
        top_left_latlon = twm.geo.move_from_latlon(north_center, 270, half_size_m)
        bottom_right_latlon = twm.geo.move_from_latlon(south_center, 90, half_size_m)
        
        lon_span = bottom_right_latlon[1] - top_left_latlon[1]
        lat_span = top_left_latlon[0] - bottom_right_latlon[0]

        frac_x = (gt_latlon[1] - top_left_latlon[1]) / lon_span if lon_span != 0 else 0.5
        frac_y = (top_left_latlon[0] - gt_latlon[0]) / lat_span if lat_span != 0 else 0.5
        gt_pixel_coords = (frac_x * TARGET_IMAGE_HEIGHT, frac_y * TARGET_IMAGE_HEIGHT)
        
        visualization_data.append({ "image": aerial_image, "gt_pixels": gt_pixel_coords, "selected_patch": (grid_row, grid_col) })

        patch_size_m = current_image_size_m / GRID_SIZE
        offset_east_m = (grid_col + 0.5) * patch_size_m - (current_image_size_m / 2.0)
        offset_north_m = -( (grid_row + 0.5) * patch_size_m - (current_image_size_m / 2.0) )
        
        temp_center = twm.geo.move_from_latlon(current_center_latlon, 90, offset_east_m)
        current_center_latlon = twm.geo.move_from_latlon(temp_center, 0, offset_north_m)
        current_image_size_m = patch_size_m

    # Calculate final GT location and load the high-res patch image
    final_mpp = current_image_size_m / TARGET_IMAGE_HEIGHT
    
    half_size_m = current_image_size_m / 2.0
    north_center = twm.geo.move_from_latlon(current_center_latlon, 0, half_size_m)
    south_center = twm.geo.move_from_latlon(current_center_latlon, 180, half_size_m)
    top_left_latlon = twm.geo.move_from_latlon(north_center, 270, half_size_m)
    bottom_right_latlon = twm.geo.move_from_latlon(south_center, 90, half_size_m)
    lon_span = bottom_right_latlon[1] - top_left_latlon[1]
    lat_span = top_left_latlon[0] - bottom_right_latlon[0]
    frac_x = (gt_latlon[1] - top_left_latlon[1]) / lon_span if lon_span != 0 else 0.5
    frac_y = (top_left_latlon[0] - gt_latlon[0]) / lat_span if lat_span != 0 else 0.5
    final_gt_pixel_coords = (frac_x * TARGET_IMAGE_HEIGHT, frac_y * TARGET_IMAGE_HEIGHT)

    final_patch_image = tileloader.load(
        latlon=current_center_latlon, bearing=0.0, meters_per_pixel=final_mpp,
        shape=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_HEIGHT)
    )

    pv_image_path = os.path.join(PV_IMAGES_PATH, f"{image_id}_undistorted.jpg")
    pv_image_raw = imageio.imread(pv_image_path)
    pv_target_w = int(TARGET_IMAGE_HEIGHT * (pv_image_raw.shape[1] / pv_image_raw.shape[0]))
    pv_image_resized = np.array(Image.fromarray(pv_image_raw).resize((pv_target_w, TARGET_IMAGE_HEIGHT)))

    num_plots = 2 + ZOOM_LEVELS
    width_ratios = [pv_target_w] + [TARGET_IMAGE_HEIGHT] * (ZOOM_LEVELS + 1)
    
    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(1, num_plots, width_ratios=width_ratios)

    ax0 = plt.subplot(gs[0])
    ax0.imshow(pv_image_resized)
    ax0.set_title("Query Street View")
    ax0.axis('off')

    for i, data in enumerate(visualization_data):
        ax = plt.subplot(gs[i+1])
        ax.imshow(data['image'])
        ax.set_title(f"Level {i} (Patch: {sequence[i]})")
        draw_grid_and_marker(ax, data['image'].shape, GRID_SIZE, data['gt_pixels'], data['selected_patch'])
        ax.set_aspect('equal', adjustable='box')

    ax_final = plt.subplot(gs[ZOOM_LEVELS + 1])
    ax_final.imshow(final_patch_image)
    ax_final.set_title(f"Final Patch (Level {ZOOM_LEVELS})")
    ax_final.plot(final_gt_pixel_coords[0], final_gt_pixel_coords[1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
    ax_final.axis('off')
    ax_final.set_aspect('equal', adjustable='box')

    plt.suptitle(f"Zoom Sequence Visualization for ID: {image_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5, w_pad=1.0)
    output_path = os.path.join(OUTPUT_DIRECTORY, f"inspected_sequence_{image_id}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main(num_samples):
    sequences_df = pd.read_csv(SEQUENCES_CSV_PATH)
    num_to_visualize = min(num_samples, len(sequences_df))
    random_samples = sequences_df.sample(n=num_to_visualize)
    
    tileloader = twm.from_yaml(os.path.join(AERIAL_DATASET_PATH, "layout.yaml"))
    tileloader = twm.WithDefault(tileloader)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    for _, row in tqdm(random_samples.iterrows(), total=len(random_samples), desc="Generating Visualizations"):
        visualize_single_id(
            image_id=row['image_id'],
            gt_latlon=np.array([row['latitude'], row['longitude']]),
            sequence=json.loads(row['sequence']),
            tileloader=tileloader
        )
        
    print(f"\n Done. Saved {num_to_visualize} visualizations to '{OUTPUT_DIRECTORY}'")

if __name__ == "__main__":
    main(10)