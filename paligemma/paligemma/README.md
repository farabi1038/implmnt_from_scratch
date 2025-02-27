# PaLiGemma Multimodal Integration

This directory contains the implementation of the PaLiGemma multimodal integration components, which combine the SigLIP vision model with the Gemma language model.

## Overview

PaLiGemma is a vision-language model that integrates:
- **Vision Component**: SigLIP Vision Transformer (in `siglip/` directory)
- **Language Component**: Gemma language model (in `gemma/` directory)
- **Cross-modal Integration**: Projection layer connecting vision and language representations

This directory specifically contains the integration components and configuration for the multimodal system.

## Components

### Configuration (`config.py`)

- `PaliGemmaConfig`: Top-level configuration class for the multimodal model
- Handles both vision and language model configurations
- Manages integration parameters like projection dimensions

### Multimodal Model (`modeling_paligemma.py`)

The multimodal integration includes:
- `PaliGemmaMultiModalProjector`: Projects vision features to text embedding space
- `PaliGemmaForConditionalGeneration`: Main model combining vision and language components
  - Processes images through vision tower
  - Merges image features with text tokens
  - Feeds combined representation to language model
  - Generates conditional text output

## Key Features

- **Token Merging**: Replaces `<image>` tokens with projected image features
- **Multimodal Context**: Enables the language model to reason about visual content
- **End-to-End Architecture**: Unified model for vision-language tasks

## Usage

```python
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from paligemma.config import PaliGemmaConfig
from gemma.config import GemmaConfig

# Define configurations
vision_config = {...}  # SigLIP vision config
text_config = {...}    # Gemma language model config

# Create PaLiGemma config
config = PaliGemmaConfig(
    vision_config=vision_config,
    text_config=text_config,
    image_token_index=256000,
    vocab_size=257152,
    projection_dim=2048
)

# Initialize model
model = PaliGemmaForConditionalGeneration(config)

# Forward pass (example)
outputs = model(
    input_ids=input_ids,           # Tokenized prompt with <image> tokens
    pixel_values=pixel_values,     # Preprocessed image tensors
    attention_mask=attention_mask  # Attention mask for input sequence
)
```

## Model Specific Notes

- Images are processed as patches and projected to the embedding dimension
- The model expects `<image>` tokens at the beginning of prompts
- Efficient generation is supported through KV caching
- The model supports both zero-shot and few-shot prompting
