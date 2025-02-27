# PaLiGemma Processors

This module contains the image and text processing components for the PaLiGemma model.

## Overview

The `processor_paligemma.py` file implements the `PaliGemmaProcessor` class which handles:

1. Image preprocessing (resizing, normalization)
2. Text tokenization with special token handling
3. Combining image and text inputs in the format expected by the PaLiGemma model

## Key Components

### PaliGemmaProcessor

This processor:
- Adds special image tokens (`<image>`) to the tokenizer
- Adds location tokens (`<loc0000>` to `<loc1023>`) for object detection
- Adds segmentation tokens (`<seg000>` to `<seg127>`) for image segmentation
- Processes images to the required size and format
- Inserts image tokens before text prompts

### Image Processing Functions

The module includes utility functions for:
- `add_image_tokens_to_prompt()`: Creates properly formatted prompts with image tokens
- `rescale()`: Rescales pixel values
- `resize()`: Resizes images to the required dimensions
- `normalize()`: Applies normalization with mean and standard deviation
- `process_images()`: Main pipeline for image processing

## Usage

```python
from processors.processor_paligemma import PaliGemmaProcessor
from transformers import AutoTokenizer
from PIL import Image

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt")

# Create processor
processor = PaliGemmaProcessor(
    tokenizer=tokenizer,
    num_image_tokens=64,
    image_size=224
)

# Process inputs
image = Image.open("example.jpg")
text_prompt = "Describe this image:"
inputs = processor(text=[text_prompt], images=[image])

# The output contains:
# - pixel_values: Tensor of processed image
# - input_ids: Tokenized text with image tokens
# - attention_mask: Attention mask for the tokens
```

## Extensions

The processor can be extended to support:
- Batch processing of multiple images and prompts
- Additional augmentation techniques
- Custom tokenization strategies
