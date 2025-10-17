from ml_collections import ConfigDict

def get_config():
    cfg = ConfigDict()

    # --- Paths ---
    cfg.paths = ConfigDict()
    cfg.paths.metadata_csv = "./ground_truth_sequences_train.csv"
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
    cfg.training.eval_interval_epochs = 5   # Evaluate on the validation set every epoch
    cfg.training.save_interval_epochs = 5   # Save a checkpoint every 5 epochs
    cfg.training.compile = True             # Enable torch.compile for a speed boost
    cfg.training.grad_clip_norm = 1.0       # Max norm for gradient clipping. Set to 0 to disable.
    cfg.training.warmup_pct = 0.05          # Percentage of total steps for LR warmup

    # --- Wandb Logging ---
    cfg.wandb = ConfigDict()
    cfg.wandb.enable = True                 # Set to False to disable logging
    cfg.wandb.project_name = "GeoZoom-CVGL"
        
    return cfg