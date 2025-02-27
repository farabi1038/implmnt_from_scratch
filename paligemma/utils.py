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
from siglip.config import PaliGemmaConfig

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Load a PaLiGemma model from a HuggingFace model path.
    
    Args:
        model_path: Path to the HuggingFace model
        device: Device to load the model onto ('cpu', 'cuda', etc.)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer) 