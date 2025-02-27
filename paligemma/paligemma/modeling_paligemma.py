# paligemma/modeling_paligemma.py
# This file implements the PaLiGemma model for conditional generation.

import torch
from torch import nn
from typing import Optional, Tuple, List
from paligemma.config import PaliGemmaConfig
from siglip.modeling_siglip_vision_model import SiglipVisionModel
from gemma.modeling_gemma import GemmaForCausalLM, KVCache

class PaliGemmaMultiModalProjector(nn.Module):
    """
    Projects vision features to the text embedding space.
    
    For example, if the vision model outputs features with dimension 768,
    and the language model expects embeddings of dimension 2048,
    this projector maps from 768 -> 2048.
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)
        
        # Example: If vision_hidden_size=768 and projection_dim=2048
        # This creates a linear layer mapping: 768 -> 2048

    def forward(self, image_features):
        # image_features shape: [Batch_Size, Num_Patches, Vision_Hidden_Size]
        # Example: [1, 196, 768] for a 224x224 image with patch size 16
        
        # Apply projection to match language model dimensions
        # Output shape: [Batch_Size, Num_Patches, Projection_Dim]
        # Example: [1, 196, 2048]
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    """
    Main PaLiGemma model for vision-language tasks.
    
    Combines:
    1. Vision model (SigLIP)
    2. Projector (to align feature spaces)
    3. Language model (Gemma)
    
    Handles the full pipeline: image encoding → feature projection → text generation
    """
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        
        # Initialize the vision model from config
        self.vision_model = SiglipVisionModel(config.vision_config)
        
        # Initialize the projection layer
        self.projector = PaliGemmaMultiModalProjector(config)
        
        # Initialize the language model from config
        self.language_model = GemmaForCausalLM(config.text_config)
        
        # Set special token index for image tokens
        # This is the token ID that will be replaced with image features
        self.image_token_index = config.image_token_index
    
    def get_input_embeddings(self):
        """Return the embeddings of the language model."""
        return self.language_model.get_input_embeddings()
    
    def tie_weights(self):
        """Tie weights between input embeddings and LM head."""
        self.language_model.tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """
        Forward pass of the multimodal model.
        
        Args:
            input_ids: Token IDs from the tokenizer (includes image tokens)
            pixel_values: Processed image tensors
            attention_mask: Mask for the input sequence
            position_ids: Position indices for tokens
            inputs_embeds: Pre-computed input embeddings (optional)
            labels: Target labels for training
            kv_cache: Key-value cache for efficient generation
            
        Example flow:
            1. Input: 
               - input_ids = tensor([[256000, 1024, 2048, ...]])  # 256000 is image token
               - pixel_values = tensor of shape [1, 3, 224, 224]  # Batch 1, RGB, 224x224
               
            2. Process:
               - Vision model processes image → image features
               - Replace image tokens with projected image features
               - Language model generates output based on combined representation
        """
        
        # Process the input image with the vision model if provided
        if pixel_values is not None:
            # pixel_values shape: [Batch_Size, 3, Height, Width] 
            # Example: [1, 3, 224, 224]
            
            # Get image features from vision model
            # vision_outputs shape: [Batch_Size, Num_Patches, Vision_Hidden_Size]
            # Example: [1, 196, 768] for 224x224 image with 16x16 patches
            vision_outputs = self.vision_model(pixel_values)
            
            # Project vision features to match language model dimensions
            # projected_features shape: [Batch_Size, Num_Patches, Projection_Dim]
            # Example: [1, 196, 2048]
            projected_features = self.projector(vision_outputs)

        # If inputs_embeds is not provided, create embeddings from input_ids
        if inputs_embeds is None and input_ids is not None:
            # Get the embedding layer from the language model
            embedding_layer = self.get_input_embeddings()
            
            # Convert input_ids to embeddings
            # input_ids shape: [Batch_Size, Seq_Len]
            # inputs_embeds shape: [Batch_Size, Seq_Len, Hidden_Size]
            inputs_embeds = embedding_layer(input_ids)
            
            # If we have vision features, replace image token embeddings with them
            if pixel_values is not None:
                # Find where the image tokens are in the sequence
                # Example: If input_ids = [[256000, 1024, ...]], 
                # image_token_mask will be [[True, False, ...]]
                image_token_mask = input_ids == self.image_token_index
                
                # Get the indices where image tokens appear
                batch_indices, token_indices = torch.where(image_token_mask)
                
                # For each image token found, replace its embedding with vision features
                for batch_idx, token_idx in zip(batch_indices, token_indices):
                    # Replace a single token embedding with a sequence of patch embeddings
                    # This effectively expands the sequence length
                    
                    # Before: inputs_embeds[batch_idx] shape: [Seq_Len, Hidden_Size]
                    # After: inputs_embeds[batch_idx] shape: 
                    # [Seq_Len + Num_Patches - 1, Hidden_Size]
                    
                    # Example: If Seq_Len=10, Num_Patches=196:
                    # Before replacement: [10, 2048]
                    # After replacement: [10+196-1, 2048] = [205, 2048]
                    
                    # First part before image token
                    prefix = inputs_embeds[batch_idx, :token_idx]
                    
                    # Image features to insert
                    image_features = projected_features[batch_idx]
                    
                    # Last part after image token
                    suffix = inputs_embeds[batch_idx, token_idx+1:]
                    
                    # Concatenate the three parts
                    new_embeds = torch.cat([prefix, image_features, suffix], dim=0)
                    
                    # Update the embeddings for this batch item
                    inputs_embeds[batch_idx] = new_embeds
        
        # Forward pass through the language model with the combined embeddings
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        
        # Return language model outputs (including logits)
        return outputs 