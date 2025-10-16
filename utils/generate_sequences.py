import os
import sys
import numpy as np
import pandas as pd
import tiledwebmaps as twm
from tqdm import tqdm

# --- Configuration ---
PV_METADATA_PATH = "/local/scratch2/cross_view_data/filtered_street_v2/image_metadata_filtered_processed_full.parquet"
AERIAL_DATASET_PATH = "/local/scratch2/cross_view_data/sat"
OUTPUT_CSV_PATH = "./ground_truth_sequences.csv"

GEOGRAPHIC_CENTER_LATLON = np.array([38.8936, -77.0116])
REGION_BOUNDS_METERS = [700.0, 2700.0, -1000.0, 1000.0]

ZOOM_LEVELS = 3
GRID_SIZE = 4
TARGET_IMAGE_HEIGHT = 224
# --- End Configuration ---

def main():
    ## Step 1: Load metadata
    print(f"Loading metadata from '{PV_METADATA_PATH}'...")
    df = pd.read_parquet(PV_METADATA_PATH)
    print(f"Found metadata for {len(df)} total images.")

    ## Step 2: Define region boundaries in lat/lon
    min_east_m, max_east_m = REGION_BOUNDS_METERS[0], REGION_BOUNDS_METERS[1]
    min_north_m, max_north_m = REGION_BOUNDS_METERS[2], REGION_BOUNDS_METERS[3]

    top_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 0, max_north_m)
    bottom_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 180, -min_north_m)
    region_top_left = twm.geo.move_from_latlon(top_center, 270, -min_east_m)
    region_bottom_right = twm.geo.move_from_latlon(bottom_center, 90, max_east_m)
    min_lon, max_lat = region_top_left[1], region_top_left[0]
    max_lon, min_lat = region_bottom_right[1], region_bottom_right[0]

    ## Step 3: Filter DataFrame
    valid_samples_df = df[
        (df['geometry.lat'] >= min_lat) & (df['geometry.lat'] <= max_lat) &
        (df['geometry.long'] >= min_lon) & (df['geometry.long'] <= max_lon)
    ].copy()
    print(f"Found {len(valid_samples_df)} images within the defined geographic region. Processing...")

    ## Step 4: Calculate initial search canvas parameters
    center_east_m = (min_east_m + max_east_m) / 2.0
    center_north_m = (min_north_m + max_north_m) / 2.0
    temp_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 90, center_east_m)
    initial_center_latlon = twm.geo.move_from_latlon(temp_center, 0, center_north_m)
    initial_image_size_meters = max_east_m - min_east_m

    ## Step 5: Iterate and generate sequences
    results = []
    for index, row in tqdm(valid_samples_df.iterrows(), total=len(valid_samples_df), desc="Generating Sequences"):
        gt_latlon = np.array([row['geometry.lat'], row['geometry.long']])
        image_id = row['id']
        
        ground_truth_sequence = []
        current_center_latlon = initial_center_latlon
        current_image_size_meters = initial_image_size_meters

        for k in range(ZOOM_LEVELS):
            half_size_m = current_image_size_meters / 2.0
            north_center = twm.geo.move_from_latlon(current_center_latlon, 0, half_size_m)
            south_center = twm.geo.move_from_latlon(current_center_latlon, 180, half_size_m)
            top_left_latlon = twm.geo.move_from_latlon(north_center, 270, half_size_m)
            bottom_right_latlon = twm.geo.move_from_latlon(south_center, 90, half_size_m)

            total_lon_span = bottom_right_latlon[1] - top_left_latlon[1]
            total_lat_span = top_left_latlon[0] - bottom_right_latlon[0]
            
            frac_x = (gt_latlon[1] - top_left_latlon[1]) / total_lon_span
            frac_y = (top_left_latlon[0] - gt_latlon[0]) / total_lat_span
            
            pixel_x = frac_x * TARGET_IMAGE_HEIGHT
            pixel_y = frac_y * TARGET_IMAGE_HEIGHT
            
            grid_row = int(pixel_y // (TARGET_IMAGE_HEIGHT / GRID_SIZE))
            grid_col = int(pixel_x // (TARGET_IMAGE_HEIGHT / GRID_SIZE))

            # Clamp values to valid grid range for edge cases 
            grid_row = max(0, min(grid_row, GRID_SIZE - 1))
            grid_col = max(0, min(grid_col, GRID_SIZE - 1))
            
            patch_index = grid_row * GRID_SIZE + grid_col
            ground_truth_sequence.append(patch_index)

            patch_size_m = current_image_size_meters / GRID_SIZE
            offset_east_m = (grid_col + 0.5) * patch_size_m
            offset_south_m = (grid_row + 0.5) * patch_size_m
            temp_point = twm.geo.move_from_latlon(top_left_latlon, 90, offset_east_m)
            current_center_latlon = twm.geo.move_from_latlon(temp_point, 180, offset_south_m)
            current_image_size_meters = patch_size_m


        results.append({
            'image_id': image_id,
            'latitude': gt_latlon[0],
            'longitude': gt_latlon[1],
            'sequence': str(ground_truth_sequence)
        })

    ## Step 6: Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nDone. Saved {len(results_df)} sequences to '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()