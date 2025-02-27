# run_paligemma.py
# This script demonstrates how to use the full PaLiGemma model.

import torch
from PIL import Image
from transformers import AutoTokenizer
from siglip.config import PaliGemmaConfig
from paligemma.config import GemmaConfig
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, KVCache
from processors.processor_paligemma import PaliGemmaProcessor

def main():
    # Create vision configuration
    vision_config = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_channels": 3,
        "image_size": 224,
        "patch_size": 16,
        "layer_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5]
    }
    
    # Create text configuration
    text_config = {
        "vocab_size": 257152,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
    }
    
    # Create PaLiGemma configuration
    config = PaliGemmaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=0
    )
    
    # Load a tokenizer (you'll need to use an appropriate tokenizer for your model)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt")
    
    # Initialize the processor
    processor = PaliGemmaProcessor(tokenizer=tokenizer, config=config.vision_config)
    
    # Initialize the model
    model = PaliGemmaForConditionalGeneration(config)
    
    # Create a sample image
    dummy_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
    
    # Create a sample text prompt
    text_prompt = "Describe this image:"
    
    # Process the inputs
    inputs = processor(text=[text_prompt], images=[dummy_image])
    
    # Create a KV cache for efficient generation
    kv_cache = KVCache()
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            kv_cache=kv_cache
        )
    
    # Print the output shape
    print("Logits shape:", output["logits"].shape)
    
    # Demonstration of the generation loop (simplified)
    generated_ids = inputs["input_ids"].clone()
    for i in range(20):  # Generate 20 new tokens
        # Forward pass with the current sequence
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids[:, -1:],  # Only the last token
                pixel_values=inputs["pixel_values"],
                attention_mask=torch.ones((1, 1), device=generated_ids.device),
                kv_cache=kv_cache
            )
        
        # Get the next token ID
        next_token_logits = outputs["logits"][:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token to the sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # Decode the generated IDs
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nGenerated text:", generated_text)


if __name__ == "__main__":
    main() 