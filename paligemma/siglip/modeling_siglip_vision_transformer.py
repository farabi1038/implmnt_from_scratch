# modeling_siglip_vision_transformer.py
# This module assembles the Vision Transformer backbone by combining embeddings and the encoder stack.

import torch
import torch.nn as nn
from config import SiglipVisionConfig
from modeling_siglip_vision_embeddings import SiglipVisionEmbeddings
from modeling_siglip_vision_encoder import SiglipEncoder

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Create patch embeddings.
        self.embeddings = SiglipVisionEmbeddings(config)
        # Create the encoder by stacking multiple transformer encoder layers.
        self.encoder = SiglipEncoder(config)
        # Final layer normalization applied after the encoder.
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Convert input image to patch embeddings.
        # Input shape: [Batch_Size, Channels, Height, Width]
        # Output shape: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        # Pass the tokens through the encoder.
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        # Apply final layer normalization.
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
