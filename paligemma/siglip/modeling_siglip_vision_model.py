# modeling_siglip_vision_model.py
# This is the top-level model that wraps the Vision Transformer backbone.

import torch
import torch.nn as nn
from config import SiglipVisionConfig
from modeling_siglip_vision_transformer import SiglipVisionTransformer

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Instantiate the Vision Transformer backbone.
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Input: [Batch_Size, Channels, Height, Width]
        # Output: [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
