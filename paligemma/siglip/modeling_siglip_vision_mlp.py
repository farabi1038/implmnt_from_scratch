# modeling_siglip_vision_mlp.py
# This module implements the two-layer feedforward network (MLP) used in the Transformer encoder.

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SiglipVisionConfig

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # First linear layer expands the dimension.
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second linear layer projects back to the hidden size.
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply first linear transformation.
        hidden_states = self.fc1(hidden_states)  # [B, N, Intermediate_Size]
        # Apply GELU activation.
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        # Apply second linear transformation.
        hidden_states = self.fc2(hidden_states)  # [B, N, Hidden_Size]
        return hidden_states
