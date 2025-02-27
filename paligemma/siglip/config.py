# config.py
# This file defines the configuration class for the Vision Transformer model.

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,              # Embedding dimension for each patch token.
        intermediate_size=3072,       # Size for the hidden layer in the MLP.
        num_hidden_layers=12,         # Number of Transformer encoder layers.
        num_attention_heads=12,       # Number of attention heads in each encoder layer.
        num_channels=3,               # Number of image channels (e.g., 3 for RGB).
        image_size=224,               # Image height and width (assumes square images).
        patch_size=16,                # Size of each patch (e.g., 16x16).
        layer_norm_eps=1e-6,          # Epsilon for layer normalization.
        attention_dropout=0.0,        # Dropout rate for attention.
        num_image_tokens: int = None, # Optional override for number of image tokens.
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class PaliGemmaConfig(SiglipVisionConfig):
    """Configuration class for PaLiGemma model."""
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=64,           # Number of image tokens used in processor.
        image_mean=[0.5, 0.5, 0.5],    # Mean values for image normalization.
        image_std=[0.5, 0.5, 0.5],     # Standard deviation values for image normalization.
        **kwargs
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            num_image_tokens=num_image_tokens,
            **kwargs
        )
        self.image_mean = image_mean
        self.image_std = image_std
