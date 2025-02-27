# PaLiGemma Model

This directory contains the implementation of the PaLiGemma multimodal model which combines a vision model (SigLIP) with a language model (Gemma).

## Overview

PaLiGemma is a vision-language model based on:
- **Vision Component**: SigLIP Vision Transformer for image understanding
- **Language Component**: Gemma for text generation
- **Cross-modal integration**: Projection layer connecting vision and language components

## Model Architecture

### Configurations

- `config.py`: Defines `GemmaConfig` for configuring the language model components.
- Configuration parameters include model dimensions, attention heads, layer counts, etc.

### Language Model (`modeling_gemma.py`)

The language model implementation includes:
- `KVCache`: For efficient autoregressive generation
- `GemmaRMSNorm`: Root Mean Square Layer Normalization
- `GemmaRotaryEmbedding`: Rotary position embeddings (RoPE)
- `GemmaMLP`: MLP block in transformer layers
- `GemmaAttention`: Multi-head attention with grouped-query attention
- `GemmaDecoderLayer`: Single transformer decoder layer
- `GemmaModel`: Full sequence model
- `GemmaForCausalLM`: Language model with LM head

### Multimodal Model (`modeling_paligemma.py`)

The multimodal integration:
- `PaliGemmaMultiModalProjector`: Projects vision features to text embedding space
- `PaliGemmaForConditionalGeneration`: Main model combining vision and language components
  - Processes images through vision tower
  - Merges image features with text tokens
  - Feeds combined representation to language model
  - Generates conditional text output

## Key Features

- **Token Merging**: Replaces `<image>` tokens with projected image features
- **Efficient Generation**: Uses key-value caching for fast autoregressive generation
- **Rotary Embeddings**: Uses RoPE for better handling of positional information
- **Grouped-Query Attention**: Optimizes attention computation for large models

## Usage

```python
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from siglip.config import PaliGemmaConfig
from paligemma.config import GemmaConfig

# Define configurations
vision_config = {...}  # SigLIP vision config
text_config = {...}  # Gemma language model config

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
    input_ids=input_ids,  # Tokenized prompt with <image> tokens
    pixel_values=pixel_values,  # Preprocessed image tensors
    attention_mask=attention_mask  # Attention mask for input sequence
)
```

## Model Specific Notes

- Images are processed as patches and projected to the embedding dimension
- The model expects `<image>` tokens at the beginning of prompts
- Efficient generation is supported through KV caching
- The model supports both zero-shot and few-shot prompting
