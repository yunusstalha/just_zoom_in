from ml_collections import ConfigDict

def get_config():
    cfg = ConfigDict()

    # --- Paths ---
    cfg.paths = ConfigDict()
    cfg.paths.metadata_csv = "./ground_truth_sequences_val.csv"
    cfg.paths.ground_root = "/local/scratch2/cross_view_data/filtered_street_v2/images"
    cfg.paths.tile_layout = "/local/scratch2/cross_view_data/sat/layout.yaml"
    cfg.paths.log_dir = "./logs"
    cfg.paths.checkpoint_dir = "./checkpoints"

    # --- Data ---
    cfg.data = ConfigDict()
    cfg.data.grid_size = 4
    cfg.data.sequence_length = 3
    cfg.data.target_image_size = 224
    cfg.data.geographic_center_latlon = [38.8936, -77.0116]
    cfg.data.region_bounds_meters = [700.0, 2700.0, -1000.0, 1000.0]

    # --- Model Architecture ---
    cfg.model = ConfigDict()
    cfg.model.encoder_name = "facebook/dinov2-base"
    cfg.model.freeze_backbone = True
    cfg.model.decoder_num_heads = 8
    cfg.model.decoder_num_layers = 6

    # --- Training ---
    cfg.training = ConfigDict()
    cfg.training.batch_size = 128
    cfg.training.learning_rate = 3e-4
    cfg.training.num_epochs = 50
    cfg.training.log_interval = 20  # Log metrics every 20 steps
    cfg.training.project_name = "GeoZoom-CVGL"
    cfg.training.run_name = "dinov2-base-run-1" # Give a name for this specific run

    return cfg