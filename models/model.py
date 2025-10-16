import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable
from .encoder import VisionTransformerEncoder
from .decoder import AutoRegressiveDecoder

class GeoLocalizationModel(nn.Module):
    """
    The main model that integrates the dual-stream encoder and the
    auto-regressive decoder for sequential cross-view geo-localization.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Initialize the dual-stream vision encoder
        self.encoder = VisionTransformerEncoder(
            model_name=config.model.encoder_name,
            freeze_backbone=config.model.freeze_backbone
        )
        
        # 2. Initialize the auto-regressive decoder
        # The decoder's embedding dimension must match the encoder's output dimension.
        self.decoder = AutoRegressiveDecoder(
            config=config,
            embed_dim=self.encoder.output_dim
        )

    def forward(self, ground_images: torch.Tensor, satellite_sequence: torch.Tensor, target_sequence: torch.Tensor):
        """
        Defines the forward pass for training with teacher-forcing.

        Args:
            ground_images (torch.Tensor): A batch of ground-view query images.
                                          Shape: (B, C, H, W).
            satellite_sequence (torch.Tensor): The ground-truth sequence of satellite images.
                                               Shape: (B, S, C, H, W).
            target_sequence (torch.Tensor): The ground-truth sequence of action indices.
                                            Shape: (B, S).

        Returns:
            torch.Tensor: The output logits for each step in the sequence.
                          Shape: (B, S, vocab_size).
        """
        # 1. Get image features from the encoder
        features = self.encoder(
            ground_images=ground_images,
            satellite_images=satellite_sequence
        )
        
        # 2. Pass the relevant features to the decoder
        # The decoder uses the global ground feature as the initial state and the
        # sequence of global satellite features as observations.
        logits = self.decoder(
            ground_global_feature=features["ground_global"],
            satellite_sequence_features=features["satellite_global"],
            target_sequence=target_sequence
        )
        
        return logits

    @torch.inference_mode()
    def generate(self, ground_images: torch.Tensor, get_next_satellite_image: Callable):
        """
        Generates a sequence of zoom actions auto-regressively for inference.

        Note: This is an illustrative implementation. The `get_next_satellite_image`
        function is a placeholder for the logic that would exist in `eval.py` to
        load the next satellite patch based on the predicted action.

        Args:
            ground_images (torch.Tensor): A batch of ground-view query images.
                                          Shape: (B, C, H, W).
            get_next_satellite_image (Callable): A function that takes a batch of
                                                 action indices and the current search
                                                 state, and returns the next batch of
                                                 satellite images.

        Returns:
            torch.Tensor: The generated sequence of action indices.
                          Shape: (B, S).
        """
        B = ground_images.shape[0]
        S = self.config.data.sequence_length
        device = ground_images.device

        # 1. Encode the ground image once at the beginning.
        ground_global_feature, _ = self.encoder._process_batch(ground_images)

        # Initialize lists to store the history of actions and satellite features
        actions_history = []
        satellite_features_history = []
        
        # The initial satellite image is the full, zoomed-out map
        next_satellite_image = get_next_satellite_image(actions=None)

        for step in range(S):
            # 2. Encode the current satellite image
            current_sat_feature, _ = self.encoder._process_batch(next_satellite_image)
            satellite_features_history.append(current_sat_feature)

            # 3. Prepare inputs for the decoder by padding history to full length
            # This is a simple way to reuse the `forward` logic. More optimized
            # methods using KV caching are possible but more complex.
            sat_feats_tensor = torch.stack(satellite_features_history, dim=1)
            
            # Use dummy actions for the parts of the sequence we are predicting
            dummy_actions = torch.zeros((B, S), dtype=torch.long, device=device)
            if actions_history:
                actions_tensor = torch.stack(actions_history, dim=1)
                dummy_actions[:, :step] = actions_tensor

            # 4. Call the decoder with the current history
            logits_full_sequence = self.decoder(
                ground_global_feature=ground_global_feature,
                satellite_sequence_features=F.pad(sat_feats_tensor, (0, 0, 0, S - (step + 1))),
                target_sequence=dummy_actions
            )

            # 5. Get the logits for the current step and predict the next action
            logits_current_step = logits_full_sequence[:, step, :]
            next_action = torch.argmax(logits_current_step, dim=-1) # Shape: (B,)
            
            actions_history.append(next_action)

            # 6. Use the predicted action to get the next satellite image (if not the last step)
            if step < S - 1:
                next_satellite_image = get_next_satellite_image(actions=next_action)

        return torch.stack(actions_history, dim=1)