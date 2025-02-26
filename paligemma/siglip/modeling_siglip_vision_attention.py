# modeling_siglip_vision_attention.py
# This module implements multi-head self-attention.

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SiglipVisionConfig

class SiglipAttention(nn.Module):
    """Multi-headed self-attention module."""
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size            # Total embedding dimension.
        self.num_heads = config.num_attention_heads      # Number of attention heads.
        self.head_dim = self.embed_dim // self.num_heads   # Dimension per head.
        self.scale = self.head_dim ** -0.5                # Scaling factor: 1/sqrt(head_dim).
        self.dropout = config.attention_dropout           # Dropout rate for attention weights.

        # Linear layers to project input into queries, keys, and values.
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # Final output projection.
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()

        # Compute query, key, and value projections.
        query_states = self.q_proj(hidden_states)  # [B, N, E]
        key_states = self.k_proj(hidden_states)      # [B, N, E]
        value_states = self.v_proj(hidden_states)      # [B, N, E]

        # Reshape and transpose to split into multiple heads:
        # New shape: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        # Expected shape: [B, Num_Heads, Num_Patches, Num_Patches]

        # Softmax to obtain attention weights.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout.
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute weighted sum of the value vectors.
        attn_output = torch.matmul(attn_weights, value_states)  # [B, Num_Heads, Num_Patches, Head_Dim]

        # Concatenate heads: transpose and reshape to [B, Num_Patches, Embed_Dim].
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # Apply the final projection.
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
