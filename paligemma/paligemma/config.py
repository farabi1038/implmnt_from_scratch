# paligemma/config.py
# This file defines the configuration classes for the Gemma model.

import torch

class GemmaConfig:
    """Configuration class for the Gemma language model."""
    
    def __init__(
        self,
        vocab_size=257152,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=28,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        
        # Optional parameters for integration with vision model
        self.num_image_tokens = kwargs.get("num_image_tokens", None) 