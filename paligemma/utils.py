# utils.py
# This file contains utility functions for loading models.

from typing import Tuple
import os
import glob
import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from paligemma.config import PaliGemmaConfig

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Load a PaLiGemma model from a HuggingFace model path.
    
    Args:
        model_path: Path to the model directory (local or HF hub ID)
        device: Device to load the model on ("cpu", "cuda", "mps")
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        model, tokenizer = load_hf_model(
            "google/paligemma-3b-pt", 
            device="cuda"
        )
    """
    # Load the tokenizer from the model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get all safetensors files in the model directory
    # Safetensors is a more secure format for storing model weights
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    
    # Dictionary to store loaded tensors
    tensors = {}
    
    # Load tensors from each file
    for tensor_file in safetensors_files:
        with safe_open(tensor_file, framework="pt", device=device) as f:
            # Load each tensor from the file and add it to the dictionary
            # Example: {"model.embed_tokens.weight": tensor([...]), ...}
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config file
    with open(os.path.join(model_path, "config.json"), "r") as f:
        # Parse the JSON config file
        model_config_file = json.load(f)
        
        # Create a PaliGemmaConfig object from the loaded config
        # This contains nested configs for vision and language models
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    # This initializes all parameters with random values
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    # This replaces random weights with pre-trained weights
    model.load_state_dict(tensors, strict=False)

    # Tie weights between input embeddings and output layer
    # Important for language models to share vocab embeddings
    model.tie_weights()

    return (model, tokenizer) 