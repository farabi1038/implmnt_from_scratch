# modeling_siglip_vision_encoder.py
# This module stacks multiple encoder layers to build the full encoder.

import torch
import torch.nn as nn
from config import SiglipVisionConfig
from modeling_siglip_vision_encoder_layer import SiglipEncoderLayer

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Create a ModuleList of encoder layers.
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds
        # Pass tokens sequentially through each encoder layer.
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
