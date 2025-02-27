# run_gemma.py
# This script demonstrates how to use the full PaLiGemma model.

import torch
from PIL import Image
from transformers import AutoTokenizer
from paligemma.config import PaliGemmaConfig
from gemma.config import GemmaConfig
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, KVCache
from processors.processor_paligemma import PaliGemmaProcessor

def main():
    # Create vision configuration
    # These parameters define the vision model architecture
    vision_config = {
        "hidden_size": 768,        # Dimension of vision features
        "intermediate_size": 3072, # Dimension of FFN intermediate layer
        "num_hidden_layers": 12,   # Number of transformer layers
        "num_attention_heads": 12, # Number of attention heads
        "num_channels": 3,         # RGB input
        "image_size": 224,         # Input image resolution
        "patch_size": 16,          # Size of image patches (16x16)
        "layer_norm_eps": 1e-6,    # Epsilon for layer norm
        "attention_dropout": 0.0,  # Dropout rate
        "image_mean": [0.5, 0.5, 0.5],  # Normalization mean
        "image_std": [0.5, 0.5, 0.5]    # Normalization std
    }
    
    # Create text configuration
    # These parameters define the language model architecture
    text_config = {
        "vocab_size": 257152,      # Size of the vocabulary 
        "hidden_size": 2048,       # Dimension of text embeddings
        "intermediate_size": 8192, # Dimension of FFN intermediate layer
        "num_hidden_layers": 18,   # Number of transformer layers
        "num_attention_heads": 8,  # Number of attention heads
        "num_key_value_heads": 1,  # Number of key/value heads (grouped-query)
        "head_dim": 256,           # Dimension per attention head
        "max_position_embeddings": 8192,  # Maximum sequence length
        "rms_norm_eps": 1e-6,      # Epsilon for RMS norm
        "rope_theta": 10000.0,     # Base for RoPE embeddings
        "attention_bias": False,   # Whether to use bias in attention
        "attention_dropout": 0.0,  # Dropout rate
    }
    
    # Create PaLiGemma configuration
    # This combines vision and language configs with integration params
    config = PaliGemmaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,         # Index to ignore in loss calculation
        image_token_index=256000,  # Token ID for image tokens
        vocab_size=257152,         # Total vocabulary size
        projection_dim=2048,       # Dimension for projecting vision to text
        hidden_size=2048,          # Main hidden dimension
        pad_token_id=0             # Token ID for padding
    )
    
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt")
    
    # Initialize the processor for preprocessing inputs
    processor = PaliGemmaProcessor(
        tokenizer=tokenizer, 
        num_image_tokens=64,  # Number of tokens per image
        image_size=224        # Image size to resize to
    )
    
    # Initialize the model with our configuration
    model = PaliGemmaForConditionalGeneration(config)
    
    # Create KV cache for efficient generation
    kv_cache = KVCache()
    
    # Create a dummy image (red 224x224 square)
    dummy_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
    
    # Create a sample text prompt
    prompt = "Describe this image in detail."
    
    # Process the inputs using the processor
    # This handles tokenization and image preprocessing
    inputs = processor(text=[prompt], images=[dummy_image])
    
    # Example of input shapes:
    # - inputs["input_ids"]: [1, N] where N is the sequence length
    # - inputs["pixel_values"]: [1, 3, 224, 224]
    # - inputs["attention_mask"]: [1, N]
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            kv_cache=kv_cache
        )
    
    # Print the output shape
    # - output["logits"] shape: [1, N, vocab_size]
    # where N is sequence length and vocab_size is the vocabulary size
    print("Logits shape:", output["logits"].shape)
    
    # Demonstration of the generation loop (simplified)
    # This shows how to generate text token by token
    generated_ids = inputs["input_ids"].clone()
    for i in range(20):  # Generate 20 new tokens
        # Forward pass with the current sequence
        # Note: We only pass the last token and use KV cache for efficiency
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids[:, -1:],  # Only the last token
                pixel_values=inputs["pixel_values"],
                attention_mask=torch.ones((1, 1), device=generated_ids.device),
                kv_cache=kv_cache
            )
        
        # Get the next token ID by taking the argmax of logits
        # This is greedy decoding (always taking most likely token)
        next_token_logits = outputs["logits"][:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token to the sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # Decode the generated IDs back to text
    # For example: "This image shows a solid red square. The color is bright red and..."
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nGenerated text:", generated_text)


if __name__ == "__main__":
    main() 