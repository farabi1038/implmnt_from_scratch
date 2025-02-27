# PaLiGemma: Vision-Language Model Implementation

This repository contains a modular implementation of the PaLiGemma vision-language model, which combines the SigLIP vision transformer with the Gemma language model for multimodal understanding and generation.

## Overview

PaLiGemma is a multimodal model capable of understanding images and generating related text, suitable for tasks such as:
- Image captioning
- Visual question answering
- Multimodal reasoning
- Visual chat

This implementation provides a modular framework with clear separation between:
- Vision components (SigLIP)
- Language components (Gemma)
- Multimodal integration (PaLiGemma)
- Preprocessing utilities

## Project Structure

```
project/
├── siglip/  # Vision model components
│   ├── config.py  # Configuration for vision model
│   └── modeling_siglip.py  # SigLIP implementation
├── paligemma/  # Language and multimodal components
│   ├── config.py  # Gemma configuration
│   ├── modeling_gemma.py  # Gemma language model
│   └── modeling_paligemma.py  # Multimodal integration
├── processors/  # Processing utilities
│   └── processor_paligemma.py  # Input processing for PaLiGemma
├── utils.py  # Utility functions for model loading
├── run_paligemma.py  # Example script for model creation
├── run_paligemma_processor.py  # Example script for processor usage
└── test_inference.py  # Inference testing script
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/paligemma-implementation.git
cd paligemma-implementation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Required dependencies include:
- PyTorch (>=1.10.0)
- transformers
- safetensors
- PIL
- numpy
- fire (for CLI functionality)

## Usage

### Quick Start

```python
import torch
from PIL import Image
from transformers import AutoTokenizer
from utils import load_hf_model
from processors.processor_paligemma import PaliGemmaProcessor

# Load model and tokenizer
model_path = "google/paligemma-3b-pt"  # Replace with actual model path
model, tokenizer = load_hf_model(model_path, device="cuda")

# Create processor
processor = PaliGemmaProcessor(
    tokenizer=tokenizer,
    num_image_tokens=model.config.vision_config.num_image_tokens,
    image_size=model.config.vision_config.image_size
)

# Process input image and text
image = Image.open("example.jpg")
prompt = "Describe this image:"
inputs = processor(text=[prompt], images=[image])

# Generate output
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"].to("cuda"),
        pixel_values=inputs["pixel_values"].to("cuda"),
        attention_mask=inputs["attention_mask"].to("cuda")
    )

# Get predicted tokens
predicted_ids = torch.argmax(outputs["logits"], dim=-1)

# Decode to text
predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
print(predicted_text)
```

### Running Inference

The `test_inference.py` script provides a command-line interface for model inference:

```bash
python test_inference.py --model_path=/path/to/model --prompt="What's happening in this image?" --image_file_path=image.jpg --max_tokens_to_generate=100 --temperature=0.8 --do_sample=True
```

## Model Loading

Models can be loaded from HuggingFace-format model directories using the `load_hf_model` function:

```python
from utils import load_hf_model

model, tokenizer = load_hf_model(
    model_path="path/to/model",
    device="cuda"  # or "cpu", "mps"
)
```

## Documentation

Each module has detailed documentation:
- [SigLIP README](siglip/README.md) - Vision model documentation
- [PaLiGemma README](paligemma/README.md) - Language and multimodal model
- [Processors README](processors/README.md) - Input processing

## License

[Insert your license information here]

## Citation

If you use this implementation in your research, please cite:

## Acknowledgments

This implementation is based on the PaLiGemma model architecture.
- SigLIP: [SigLIP paper/repo link]
- Gemma: [Gemma paper/repo link]
- PaLiGemma: [PaLiGemma paper/repo link]
