import ast
import csv
import logging

from pathlib import Path

import numpy as np
import torch
import tiledwebmaps as twm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

# Assumes a logger is configured globally, similar to nanochat's common.py
logger = logging.getLogger(__name__)

# Module-level constants are cleaner
GROUND_IMAGE_SUFFIX = "_undistorted.jpg"

def _parse_sequence(value):
    """Helper to parse a string representation of a list into a list."""
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


class ZoomDataset(Dataset):
    """
    Dataset for loading ground-level images and their corresponding sequences of satellite patches.
    """
    def __init__(self, config, transforms = None):
        self.metadata_path = Path(config.paths.metadata_csv)
        self.transforms = transforms or {}

        self.ground_root = Path(config.paths.ground_root)
        self.grid_size = config.data.grid_size
        self.target_image_size = config.data.target_image_size
        self.geographic_center = np.array(config.data.geographic_center_latlon)
        self.region_bounds = config.data.region_bounds_meters
        self.sequence_length = config.data.sequence_length

        self.tile_loader = twm.WithDefault(twm.from_yaml(config.paths.tile_layout))
        self.initial_center_latlon, self.initial_image_size = self._compute_initial_canvas()

        # --- Load data and validate ---
        self.records = self._load_metadata()
        if not self.records:
            raise ValueError(f"No samples found in {self.metadata_path}")

        # Validate that all sequences in the metadata match the configured length
        for record in self.records:
            assert len(record["sequence"]) == self.sequence_length, \
                f"Sequence length mismatch in metadata. Expected {self.sequence_length}, found {len(record['sequence'])}."

        logger.info(f"Loaded {len(self.records)} samples from {self.metadata_path}")
        logger.info(f"Sequence length: {self.sequence_length}, Grid size: {self.grid_size}")
    def _compute_initial_canvas(self):
        """Calculates the starting center and size for the satellite view."""
        min_east, max_east, min_north, max_north = self.region_bounds
        center_east = (min_east + max_east) / 2.0
        center_north = (min_north + max_north) / 2.0

        temp_center = twm.geo.move_from_latlon(self.geographic_center, 90, center_east)
        initial_center = twm.geo.move_from_latlon(temp_center, 0, center_north)
        initial_size = max_east - min_east

        return np.array(initial_center), initial_size

    def _load_metadata(self):
        """Loads records from the CSV metadata file."""
        records = []
        with self.metadata_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "image_id": row["image_id"],
                    "sequence": _parse_sequence(row["sequence"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                })
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        """Loads and returns a single sample from the dataset."""
        record = self.records[index]
        sample = {
            "ground": self._load_ground_image(record["image_id"]),
            "satellite_sequence": self._load_satellite_sequence(record["sequence"]),
            "sequence": torch.tensor(record["sequence"], dtype=torch.long),
            "meta": {
                "image_id": record["image_id"],
                "latitude": record["latitude"],
                "longitude": record["longitude"],
            },
        }
        return sample

    def _load_ground_image(self, image_id):
        """Loads a single ground-level image and applies transforms."""
        path = self.ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
        image = Image.open(path).convert("RGB")
        
        transform = self.transforms.get("ground")
        return transform(image) if transform else to_tensor(image)

    def _load_satellite_sequence(self, sequence):
        """Loads a sequence of satellite patches based on the zoom sequence."""
        patches = []
        current_center = self.initial_center_latlon
        current_size = self.initial_image_size
        transform = self.transforms.get("satellite")

        for patch_index in sequence:
            meters_per_pixel = current_size / self.target_image_size
            array = self.tile_loader.load(
                latlon=current_center,
                bearing=0.0,
                meters_per_pixel=meters_per_pixel,
                shape=(self.target_image_size, self.target_image_size),
            )
            image = Image.fromarray(array)
            patches.append(transform(image) if transform else to_tensor(image))

            # Update the center and size for the next zoom level
            patch_size = current_size / self.grid_size
            row, col = divmod(patch_index, self.grid_size)

            offset_east = (col + 0.5) * patch_size - (current_size / 2.0)
            offset_north = -((row + 0.5) * patch_size - (current_size / 2.0))

            temp_center = twm.geo.move_from_latlon(current_center, 90, offset_east)
            current_center = twm.geo.move_from_latlon(temp_center, 0, offset_north)
            current_size = patch_size

        return torch.stack(patches)
