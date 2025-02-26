# run_siglip_vision.py
# This script demonstrates how to instantiate and run the SiglipVisionModel.

import torch
from config import SiglipVisionConfig
from modeling_siglip_vision_model import SiglipVisionModel

if __name__ == "__main__":
    # Create a configuration instance with desired hyperparameters.
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0
    )

    # Instantiate the Vision Transformer model.
    model = SiglipVisionModel(config)

    # Create a dummy image tensor with shape [Batch_Size, Channels, Height, Width].
    dummy_image = torch.randn(1, 3, 224, 224)

    # Run the forward pass.
    outputs = model(dummy_image)

    # Print the output shape. Expected: [1, Num_Patches, hidden_size]
    print("Output shape:", outputs.shape)
