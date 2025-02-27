# run_processor.py
# This script demonstrates how to use the PaLiGemma processor.

import torch
from PIL import Image
from transformers import AutoTokenizer
from siglip.config import PaliGemmaConfig
from processors.processor_paligemma import PaliGemmaProcessor

if __name__ == "__main__":
    # Create a configuration instance with desired hyperparameters
    config = PaliGemmaConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=64,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5]
    )
    
    # Load a tokenizer (you'll need to use an appropriate tokenizer for your model)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt")
    
    # Instantiate the processor
    processor = PaliGemmaProcessor(tokenizer=tokenizer, config=config)
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
    
    # Create a sample text prompt
    text_prompt = "Describe this image."
    
    # Process the inputs
    inputs = processor(text=[text_prompt], images=[dummy_image])
    
    # Print the output shapes
    print("Pixel values shape:", inputs["pixel_values"].shape)
    print("Input IDs shape:", inputs["input_ids"].shape)
    print("Attention mask shape:", inputs["attention_mask"].shape)
    
    # Print the first few tokens to verify image tokens were added
    print("\nFirst few tokens:", inputs["input_ids"][0][:70])
    print("\nTokenized text:", tokenizer.decode(inputs["input_ids"][0])) 