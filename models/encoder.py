import torch
import torch.nn as nn
from transformers import AutoModel

class VisionTransformerEncoder(nn.Module):
    """
    A Vision Transformer (ViT) encoder that wraps a Hugging Face pretrained model
    to produce both global (CLS) and local (patch) embeddings.

    This module processes two streams of input: a single ground-view image and
    a sequence of satellite images per sample. The backbone weights are shared
    across both streams to learn a common feature representation.
    """
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.output_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # # Ensure the CLS token and position embeddings are also frozen
            # if hasattr(self.backbone.embeddings, 'cls_token'):
            #      self.backbone.embeddings.cls_token.requires_grad = False
            # if hasattr(self.backbone.embeddings, 'position_embeddings'):
            #      self.backbone.embeddings.position_embeddings.requires_grad = False


    def _process_batch(self, images: torch.Tensor):
        """
        Forwards a batch of images through the backbone and separates embeddings.
        """
        # The Hugging Face ViT model returns a BaseModelOutputWithPooling object
        outputs = self.backbone(pixel_values=images)
        
        # last_hidden_state has shape: (batch_size, num_patches + 1, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # The first token is the global [CLS] embedding
        global_embedding = last_hidden_state[:, 0]
        
        # The rest are patch embeddings
        local_embeddings = last_hidden_state[:, 1:]
        
        return global_embedding, local_embeddings

    def forward(self, ground_images: torch.Tensor, satellite_images: torch.Tensor):
        """
        Performs the forward pass for both ground and satellite streams.

        Args:
            ground_images: A tensor of shape (B, C, H, W).
            satellite_images: A tensor of shape (B, S, C, H, W), where S is the sequence length.

        Returns:
            A dictionary containing global and local embeddings for both streams.
        """
        # 1. Process the batch of ground images
        B_g, N_crops, C_g, H_g, W_g = ground_images.shape
        
        # Reshape to process all crops in one batch: (B, 2, C, H, W) -> (B*2, C, H, W)
        ground_images_flat = ground_images.view(B_g * N_crops, C_g, H_g, W_g)
        
        # Get flattened global embeddings of shape (B*2, D)
        ground_global_flat, _ = self._process_batch(ground_images_flat)
        
        # Reshape back and average across crops to get a single feature per sample
        # (B*2, D) -> (B, 2, D) -> (B, D)
        ground_global = ground_global_flat.view(B_g, N_crops, -1).mean(dim=1)
        ground_local = None
        # 2. Process the sequence of satellite images
        B, S, C, H, W = satellite_images.shape
        # Flatten the batch and sequence dimensions to process all at once
        satellite_images_flat = satellite_images.reshape(B * S, C, H, W)
        
        sat_global_flat, sat_local_flat = self._process_batch(satellite_images_flat)
        
        # Reshape back to restore the sequence dimension
        # (B * S, D) -> (B, S, D)
        satellite_global = sat_global_flat.view(B, S, -1)
        
        # (B * S, N, D) -> (B, S, N, D), where N is the number of patches
        num_patches = sat_local_flat.shape[1]
        satellite_local = sat_local_flat.view(B, S, num_patches, -1)
        
        return {
            "ground_global": ground_global,
            "ground_local": ground_local,
            "satellite_global": satellite_global,
            "satellite_local": satellite_local,
        }