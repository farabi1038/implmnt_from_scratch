# Gemma Language Model

This directory contains the implementation of the Gemma language model, which serves as the text generation component of the PaLiGemma multimodal system.

## Overview

Gemma is a powerful language model based on the decoder-only transformer architecture with several key enhancements:
- **Rotary Position Embeddings (RoPE)** for better positional understanding
- **Grouped-Query Attention** for efficient attention computation
- **RMS Normalization** for improved training stability
- **SwiGLU Activation** in the feed-forward networks

## Model Architecture

### Configuration

- `config.py`: Defines `GemmaConfig` for configuring the language model components
- Configuration parameters include model dimensions, attention heads, layer counts, etc.

### Core Components

The language model implementation in `modeling_gemma.py` includes:

- `KVCache`: For efficient autoregressive generation
- `GemmaRMSNorm`: Root Mean Square Layer Normalization
- `GemmaRotaryEmbedding`: Rotary position embeddings (RoPE)
- `GemmaMLP`: MLP block in transformer layers
- `GemmaAttention`: Multi-head attention with grouped-query attention
- `GemmaDecoderLayer`: Single transformer decoder layer
- `GemmaModel`: Full sequence model
- `GemmaForCausalLM`: Language model with LM head

## Key Features

- **Efficient Generation**: Uses key-value caching for fast autoregressive generation
- **Rotary Embeddings**: Uses RoPE for better handling of positional information
- **Grouped-Query Attention**: Optimizes attention computation for large models
- **Scalable Architecture**: Designed to work well across various model sizes

## Usage

```python
from gemma.config import GemmaConfig
from gemma.modeling_gemma import GemmaForCausalLM

# Create configuration
config = GemmaConfig(
    vocab_size=257152,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    max_position_embeddings=8192
)

# Initialize model
model = GemmaForCausalLM(config)

# Generate text (example)
inputs_embeds = model.get_input_embeddings()(input_ids)
outputs = model(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask
)

# Access logits
logits = outputs["logits"]
```

## Integration with PaLiGemma

The Gemma language model serves as the core text generation component of the PaLiGemma multimodal system. It processes embeddings that combine text and image features to generate contextually relevant text outputs for visual inputs.

For the full multimodal implementation, see the `paligemma/` directory that handles the integration of vision and language models. 