# modeling_siglip_vision_encoder_layer.py
# This module defines a single encoder layer, which includes self-attention and an MLP with residual connections.

import torch
import torch.nn as nn
from config import SiglipVisionConfig
from modeling_siglip_vision_attention import SiglipAttention
from modeling_siglip_vision_mlp import SiglipMLP

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Multi-head self-attention sublayer.
        self.self_attn = SiglipAttention(config)
        # Layer normalization before self-attention.
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Feedforward MLP sublayer.
        self.mlp = SiglipMLP(config)
        # Layer normalization before the MLP.
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Save input for residual connection.
        residual = hidden_states
        # Apply layer normalization.
        hidden_states = self.layer_norm1(hidden_states)
        # Apply self-attention.
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # Add the residual connection.
        hidden_states = residual + hidden_states

        # Save for second residual connection.
        residual = hidden_states
        # Layer normalization before MLP.
        hidden_states = self.layer_norm2(hidden_states)
        # Apply MLP.
        hidden_states = self.mlp(hidden_states)
        # Add residual connection.
        hidden_states = residual + hidden_states

        return hidden_states
