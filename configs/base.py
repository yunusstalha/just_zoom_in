from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.paths = ConfigDict()
    cfg.paths.metadata_csv = "./ground_truth_sequences.csv"
    cfg.paths.ground_root = "/local/scratch2/cross_view_data/filtered_street_v2/images"
    cfg.paths.tile_layout = "/local/scratch2/cross_view_data/sat/layout.yaml"
    cfg.paths.log_dir = "./logs"

    cfg.data = ConfigDict()
    cfg.data.grid_size = 4
    cfg.data.sequence_length = 3
    cfg.data.target_image_size = 224
    cfg.data.geographic_center_latlon = [38.8936, -77.0116]
    cfg.data.region_bounds_meters = [700.0, 2700.0, -1000.0, 1000.0]

    return cfg
