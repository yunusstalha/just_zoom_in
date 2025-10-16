import os
import sys
import pandas as pd
import numpy as np
import tiledwebmaps as twm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datashader as ds
from datashader import transfer_functions as tf

# Path to the Parquet file containing street-view image metadata
STREET_VIEW_METADATA_PATH = "/local/scratch2/cross_view_data/filtered_street_v2/image_metadata_filtered_processed_full.parquet"
# Path to the root directory of your satellite tile dataset
SATELLITE_DATA_PATH = "/local/scratch2/cross_view_data/sat"

# Define the geographic area in kilometers relative to the dataset's center.
# A good way to start is to visualize a small, dense area.
MAP_BOUNDS_KM = {'x': (0.7, 2.7), 'y': (-1.0, 1.0)}

TARGET_METERS_PER_PIXEL = 5.0

HEATMAP_COLORMAP = 'hot'
POINT_SPREAD_PIXELS = 2
OUTPUT_FIGURE_PATH = "./data_coverage.png"
CENTER_LATLON = (38.8936, -77.0116)

def load_gps_points(parquet_path):
    df = pd.read_parquet(parquet_path)
    required_cols = ['geometry.lat', 'geometry.long']
    
    df_gps = df[['geometry.lat', 'geometry.long']].copy()
    df_gps.rename(columns={'geometry.lat': 'lat', 'geometry.long': 'lon'}, inplace=True)
    return df_gps

def main():

    # --- Step 1: Load Street-View GPS Data ---
    print("INFO: Loading street views...")
    streetview_df = load_gps_points(STREET_VIEW_METADATA_PATH)

    # --- Step 2: Set up the Satellite Map Loader ---
    print("INFO: Initializing satellite map loader...")
    layout_path = os.path.join(SATELLITE_DATA_PATH, "layout.yaml")

    tileloader = twm.from_yaml(layout_path)
    tileloader = twm.WithDefault(tileloader) # Handles missing tiles 


    # --- Step 3: Define the Map Area and Fetch Satellite Imagery ---
    # calculate the map dimensions in meters and pixels based on configs
    map_width_km = MAP_BOUNDS_KM['x'][1] - MAP_BOUNDS_KM['x'][0]
    map_height_km = MAP_BOUNDS_KM['y'][1] - MAP_BOUNDS_KM['y'][0]
    map_width_m = map_width_km * 1000
    map_height_m = map_height_km * 1000

    output_shape_pixels = (
        int(map_height_m / TARGET_METERS_PER_PIXEL),
        int(map_width_m / TARGET_METERS_PER_PIXEL)
    )
    print(f"INFO: Fetching satellite map for a {map_width_km:.1f}km x {map_height_km:.1f}km area...")
    print(f"      Image resolution will be {output_shape_pixels[1]} x {output_shape_pixels[0]} pixels.")
    
    center_x_km = (MAP_BOUNDS_KM['x'][0] + MAP_BOUNDS_KM['x'][1]) / 2
    center_y_km = (MAP_BOUNDS_KM['y'][0] + MAP_BOUNDS_KM['y'][1]) / 2

    # Move from the true dataset center to find the geographic center of our desired map
    map_center_latlon_y = twm.geo.move_from_latlon(CENTER_LATLON, bearing=0, distance=center_y_km * 1000)
    map_center_latlon = twm.geo.move_from_latlon(CENTER_LATLON, bearing=90, distance=center_x_km * 1000)

    # This is the core tiledwebmaps function. It fetches the right tiles and stitches
    # them together into a single NumPy array image.
    background_map = tileloader.load(
        latlon=map_center_latlon,
        bearing=0.0,
        meters_per_pixel=TARGET_METERS_PER_PIXEL,
        shape=output_shape_pixels,
    )
    
    # --- Step 4: Project GPS Points onto the Map ---
    # convert (lat, lon) points into the (x, y) kilometer
    # coordinate system of map for plotting.
    print("INFO: Projecting GPS points onto the map...")
    half_height_m = map_height_m / 2
    half_width_m = map_width_m / 2

    north_edge = twm.geo.move_from_latlon(map_center_latlon, bearing=0,   distance=half_height_m)
    south_edge = twm.geo.move_from_latlon(map_center_latlon, bearing=180, distance=half_height_m)
    top_left_latlon     = twm.geo.move_from_latlon(north_edge, bearing=270, distance=half_width_m)
    bottom_right_latlon = twm.geo.move_from_latlon(south_edge, bearing=90,  distance=half_width_m)

    min_lon, max_lat = top_left_latlon[1], top_left_latlon[0]
    max_lon, min_lat = bottom_right_latlon[1], bottom_right_latlon[0]
    
    visible_mask = (
        (streetview_df['lon'] >= min_lon) & (streetview_df['lon'] <= max_lon) &
        (streetview_df['lat'] >= min_lat) & (streetview_df['lat'] <= max_lat)
    )
    visible_points_df = streetview_df[visible_mask].copy()
   
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    visible_points_df['x_km'] = MAP_BOUNDS_KM['x'][0] + ((visible_points_df['lon'] - min_lon) / lon_range) * map_width_km
    visible_points_df['y_km'] = MAP_BOUNDS_KM['y'][0] + ((visible_points_df['lat'] - min_lat) / lat_range) * map_height_km
    
    # --- Step 5: Create the Density Heatmap with Datashader ---
    print("INFO: Aggregating points into a density grid with Datashader...")
    
    canvas = ds.Canvas(
        plot_width=output_shape_pixels[1],
        plot_height=output_shape_pixels[0],
        x_range=(MAP_BOUNDS_KM['x'][0], MAP_BOUNDS_KM['x'][1]),
        y_range=(MAP_BOUNDS_KM['y'][0], MAP_BOUNDS_KM['y'][1]),
    )

    agg = canvas.points(visible_points_df, 'x_km', 'y_km')
    spread_agg = tf.spread(agg, px=POINT_SPREAD_PIXELS, name='spread')
    num_points_in_region = len(visible_points_df)
    print(f"Found {num_points_in_region} street-view points within the defined map area.")

    # --- Step 6: Combine Map and Heatmap, and Save the Figure ---
    print(f"INFO: Generating the final plot and saving to '{OUTPUT_FIGURE_PATH}'...")
    
    # Set the figure size to match the map's aspect ratio
    aspect_ratio = map_width_km / map_height_km
    fig, ax = plt.subplots(figsize=(12, 12 / aspect_ratio), dpi=200)

    ax.imshow(background_map, extent=[*MAP_BOUNDS_KM['x'], *MAP_BOUNDS_KM['y']])

    artist = ax.imshow(
        spread_agg,
        extent=[*MAP_BOUNDS_KM['x'], *MAP_BOUNDS_KM['y']],
        origin='lower', # Datashader and imshow have different y-axis conventions
        cmap=HEATMAP_COLORMAP,
        norm=LogNorm(vmin=1),
        alpha=0.8
    )

    ax.set_title(f"Street-View Data Density ({num_points_in_region} points)")
    ax.set_xlabel("Kilometers from Center (East-West)")
    ax.set_ylabel("Kilometers from Center (North-South)")
    ax.grid(True, linestyle=':', color='white', alpha=0.5)
    
    fig.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH, bbox_inches='tight')
    plt.close()
    print(f"\n All done! Map saved successfully to {OUTPUT_FIGURE_PATH}.")

if __name__ == "__main__":
    main()