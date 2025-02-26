# modeling_siglip_vision_embeddings.py
# This module defines the embeddings component that splits an image into patches and adds positional embeddings.

import torch
import torch.nn as nn
from config import SiglipVisionConfig

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size   # Final embedding dimension.
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Convolution to extract non-overlapping patches from the image.
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,   # e.g., 3 for RGB.
            out_channels=self.embed_dim,        # Project each patch to hidden_size.
            kernel_size=self.patch_size,        # Kernel size equals patch size.
            stride=self.patch_size,             # Stride equals patch size → non-overlapping.
            padding="valid"                     # No padding.
        )

        # Total number of patches is (image_size / patch_size)^2.
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Learnable positional embeddings (one per patch).
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # Buffer to store the position indices.
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),  # Shape: [1, num_patches]
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Input: [Batch_Size, Channels, Height, Width]
        patch_embeds = self.patch_embedding(pixel_values)
        # Output shape: [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        embeddings = patch_embeds.flatten(2)  # Flatten spatial dims: [Batch_Size, Embed_Dim, Num_Patches]
        embeddings = embeddings.transpose(1, 2) # Transpose to: [Batch_Size, Num_Patches, Embed_Dim]
        # Add positional embeddings.
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # Final output: [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
