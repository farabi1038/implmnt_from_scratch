# processor_paligemma.py
# This module implements the processor for the PaLiGemma model.

from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch
from paligemma.config import PaliGemmaConfig
import torchvision.transforms as T


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Prepares prompts for PaLiGemma by adding image tokens and BOS token.
    
    The input format follows PaLiGemma's expected structure:
    - A sequence of image tokens at the beginning
    - A BOS token followed by the text prompt
    - A newline character at the end
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Rescale pixel values of an image by a scale factor."""
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """Resize an image to the specified size."""
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """Normalize an image by subtracting mean and dividing by std."""
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Process a list of images for input to the model.
    
    This includes resizing, rescaling pixel values, normalizing,
    and rearranging dimensions to match the model's expected input format.
    """
    height, width = size
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    """
    Processor for PaLiGemma model that handles both image preprocessing and text tokenization.
    
    This processor:
    1. Processes images (resizing, normalization)
    2. Tokenizes text with special image tokens
    3. Creates the combined input format required by the model
    """

    IMAGE_TOKEN = "<image>"

    def __init__(
        self, 
        tokenizer, 
        num_image_tokens: int = 64,
        image_size: int = 224,
        image_mean: List[float] = [0.5, 0.5, 0.5],
        image_std: List[float] = [0.5, 0.5, 0.5]
    ):
        """
        Initialize the processor with tokenizer and image processing parameters.
        
        Args:
            tokenizer: HuggingFace tokenizer for text processing
            num_image_tokens: Number of image tokens to inject
            image_size: Size to resize images to (square)
            image_mean: Mean values for image normalization (RGB)
            image_std: Standard deviation values for normalization (RGB)
            
        Example:
            processor = PaliGemmaProcessor(
                tokenizer=AutoTokenizer.from_pretrained("google/paligemma-3b-pt"),
                num_image_tokens=64,
                image_size=224
            )
        """
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        
        # Image preprocessing pipeline
        self.image_processor = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std)
        ])
        
        # Add special tokens to tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add location and segmentation tokens
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def add_image_tokens_to_prompt(self, prompt: str, image_token_index: int = 256000) -> str:
        """
        Add image tokens to the beginning of a prompt.
        
        Args:
            prompt: Text prompt to add image tokens to
            image_token_index: Token ID for image token
            
        Returns:
            Prompt with image tokens added
            
        Example:
            Input: "Describe this image:"
            Output: "<image>Describe this image:"
            (where <image> is a special token)
        """
        # Convert the image token index to a string token representation
        # The tokenizer will convert this back to the right token ID
        image_token = self.tokenizer.decode([image_token_index])
        
        # Add the image token to the beginning of the prompt
        # Note: When tokenized, this single token will be replaced with
        # multiple image embedding tokens from the vision model
        return f"{image_token}{prompt}"

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Process a batch of images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Tensor of processed image tensors
            
        Example:
            Input: [PIL.Image.Image of size 640x480]
            Output: torch.Tensor of shape [1, 3, 224, 224]
        """
        processed_images = []
        
        for image in images:
            # Apply the image preprocessing pipeline:
            # 1. Resize to square (e.g., 224x224)
            # 2. Convert to tensor (range [0-1])
            # 3. Normalize with mean and std
            processed_image = self.image_processor(image)
            processed_images.append(processed_image)
        
        # Stack into a batch
        # Result shape: [Batch_Size, 3, Height, Width]
        # Example: [2, 3, 224, 224] for 2 images
        return torch.stack(processed_images)

    def __call__(
        self, 
        text: List[str], 
        images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of text prompts and images.
        
        Args:
            text: List of text prompts
            images: List of PIL images
            
        Returns:
            Dictionary with processed inputs:
            - input_ids: Token IDs for text (with image tokens)
            - pixel_values: Processed image tensors
            - attention_mask: Attention mask for input sequence
            
        Example:
            Input:
              text=["Describe this image:"]
              images=[PIL.Image.Image of size 640x480]
            
            Output:
              {
                "input_ids": tensor([[256000, 1024, 2048, ...]]),
                "pixel_values": tensor of shape [1, 3, 224, 224],
                "attention_mask": tensor([[1, 1, 1, ...]])
              }
        """
        if len(text) != len(images):
            raise ValueError(
                f"Number of text prompts ({len(text)}) must match "
                f"number of images ({len(images)})"
            )
        
        # Add image tokens to each prompt
        # This is typically a special token at the beginning of the prompt
        processed_text = [self.add_image_tokens_to_prompt(t) for t in text]
        
        # Tokenize the text prompts
        # This converts text to token IDs that the model understands
        encoded_text = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Process the images
        # This resizes, normalizes, and converts to tensors
        pixel_values = self.process_images(images)
        
        # Combine into a single dictionary of inputs
        inputs = {
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "pixel_values": pixel_values
        }
        
        return inputs 