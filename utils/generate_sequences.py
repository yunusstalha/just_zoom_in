import os
import sys
import numpy as np
import pandas as pd
import tiledwebmaps as twm
from tqdm import tqdm

PV_METADATA_PATH = "/local/scratch2/cross_view_data/filtered_street_v2/image_metadata_filtered_processed_full.parquet"
AERIAL_DATASET_PATH = "/local/scratch2/cross_view_data/sat"

# --- Output Paths ---
OUTPUT_TRAIN_CSV_PATH = "./ground_truth_sequences_train.csv"
OUTPUT_VAL_CSV_PATH = "./ground_truth_sequences_val.csv"

# --- Data Split Configuration ---
VAL_SET_FRACTION = 0.1  # Use 10% of the data for the validation set
RANDOM_SEED = 42        # A fixed seed for reproducible splits

# --- Sequence Generation Parameters ---
GEOGRAPHIC_CENTER_LATLON = np.array([38.8936, -77.0116])
REGION_BOUNDS_METERS = [700.0, 2700.0, -1000.0, 1000.0]
ZOOM_LEVELS = 3
GRID_SIZE = 4
TARGET_IMAGE_HEIGHT = 224 # This is used for pixel projection calculations


def generate_sequences_for_df(df, initial_center_latlon, initial_image_size_meters, desc=""):
    """Processes a DataFrame to generate ground-truth zoom sequences for each row."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        gt_latlon = np.array([row['geometry.lat'], row['geometry.long']])
        image_id = row['id']
        
        ground_truth_sequence = []
        current_center_latlon = initial_center_latlon
        current_image_size_meters = initial_image_size_meters

        for _ in range(ZOOM_LEVELS):
            # Calculate the geographic corners of the current view
            half_size_m = current_image_size_meters / 2.0
            north_center = twm.geo.move_from_latlon(current_center_latlon, 0, half_size_m)
            south_center = twm.geo.move_from_latlon(current_center_latlon, 180, half_size_m)
            top_left_latlon = twm.geo.move_from_latlon(north_center, 270, half_size_m)
            bottom_right_latlon = twm.geo.move_from_latlon(south_center, 90, half_size_m)

            # Convert GT lat/lon to a fractional position (0-1) within the current view
            lon_span = bottom_right_latlon[1] - top_left_latlon[1]
            lat_span = top_left_latlon[0] - bottom_right_latlon[0]
            frac_x = (gt_latlon[1] - top_left_latlon[1]) / lon_span if lon_span != 0 else 0.5
            frac_y = (top_left_latlon[0] - gt_latlon[0]) / lat_span if lat_span != 0 else 0.5
            
            # Determine which grid cell the GT point falls into
            grid_row = int(frac_y * GRID_SIZE)
            grid_col = int(frac_x * GRID_SIZE)
            
            # Clamp values to handle edge cases where point is exactly on the boundary
            grid_row = max(0, min(grid_row, GRID_SIZE - 1))
            grid_col = max(0, min(grid_col, GRID_SIZE - 1))
            
            patch_index = grid_row * GRID_SIZE + grid_col
            ground_truth_sequence.append(patch_index)

            # Calculate the center of the selected patch to be the center for the next zoom level
            patch_size_m = current_image_size_meters / GRID_SIZE
            offset_east_m = (grid_col + 0.5) * patch_size_m - half_size_m
            offset_north_m = -((grid_row + 0.5) * patch_size_m - half_size_m)
            
            temp_center = twm.geo.move_from_latlon(current_center_latlon, 90, offset_east_m)
            current_center_latlon = twm.geo.move_from_latlon(temp_center, 0, offset_north_m)
            current_image_size_meters = patch_size_m

        results.append({
            'image_id': image_id,
            'latitude': gt_latlon[0],
            'longitude': gt_latlon[1],
            'sequence': str(ground_truth_sequence)
        })
    return pd.DataFrame(results)


def main():
    # Step 1: Load and filter metadata to the region of interest
    print(f"Loading metadata from '{PV_METADATA_PATH}'...")
    df = pd.read_parquet(PV_METADATA_PATH)
    
    min_east_m, max_east_m, min_north_m, max_north_m = REGION_BOUNDS_METERS
    top_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 0, max_north_m)
    bottom_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 180, -min_north_m)
    region_top_left = twm.geo.move_from_latlon(top_center, 270, -min_east_m)
    region_bottom_right = twm.geo.move_from_latlon(bottom_center, 90, max_east_m)
    min_lon, max_lat = region_top_left[1], region_top_left[0]
    max_lon, min_lat = region_bottom_right[1], region_bottom_right[0]

    valid_samples_df = df[
        (df['geometry.lat'] >= min_lat) & (df['geometry.lat'] <= max_lat) &
        (df['geometry.long'] >= min_lon) & (df['geometry.long'] <= max_lon)
    ].copy()
    print(f"Found {len(valid_samples_df)} images within the defined geographic region.")

    # Step 2: Split the filtered data into training and validation sets
    val_df = valid_samples_df.sample(frac=VAL_SET_FRACTION, random_state=RANDOM_SEED)
    train_df = valid_samples_df.drop(val_df.index)
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # Step 3: Calculate initial search canvas parameters
    center_east_m = (min_east_m + max_east_m) / 2.0
    center_north_m = (min_north_m + max_north_m) / 2.0
    temp_center = twm.geo.move_from_latlon(GEOGRAPHIC_CENTER_LATLON, 90, center_east_m)
    initial_center_latlon = twm.geo.move_from_latlon(temp_center, 0, center_north_m)
    initial_image_size_meters = max_east_m - min_east_m

    # Step 4: Process the training set and save to CSV
    train_results_df = generate_sequences_for_df(
        train_df, initial_center_latlon, initial_image_size_meters, desc="Generating Train Sequences"
    )
    train_results_df.to_csv(OUTPUT_TRAIN_CSV_PATH, index=False)
    print(f"Saved {len(train_results_df)} training sequences to '{OUTPUT_TRAIN_CSV_PATH}'")

    # Step 5: Process the validation set and save to CSV
    val_results_df = generate_sequences_for_df(
        val_df, initial_center_latlon, initial_image_size_meters, desc="Generating Val Sequences  "
    )
    val_results_df.to_csv(OUTPUT_VAL_CSV_PATH, index=False)
    print(f"Saved {len(val_results_df)} validation sequences to '{OUTPUT_VAL_CSV_PATH}'")
    
    print("\nAll done.")

if __name__ == "__main__":
    main()