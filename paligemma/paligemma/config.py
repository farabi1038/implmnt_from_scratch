# paligemma/config.py
# This file defines the configuration class for the PaLiGemma multimodal model.

from siglip.config import SiglipVisionConfig
from gemma.config import GemmaConfig

class PaliGemmaConfig:
    """Configuration class for the PaLiGemma multimodal model."""
    
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        # Process vision config
        if isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Process text config
        if isinstance(text_config, dict):
            self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        else:
            self.text_config = text_config
            if pad_token_id is not None:
                self.text_config.pad_token_id = pad_token_id

        # Update configs
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim 