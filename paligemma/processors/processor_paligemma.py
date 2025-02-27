# processor_paligemma.py
# This module implements the processor for the PaLiGemma model.

from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch
from paligemma.config import PaliGemmaConfig


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

    def __init__(self, tokenizer, config=None, num_image_tokens=None, image_size=None):
        """
        Initialize the processor with a tokenizer and configuration.
        
        Args:
            tokenizer: A tokenizer for processing text inputs
            config: PaliGemmaConfig object with model parameters
            num_image_tokens: Optional override for the number of image tokens
            image_size: Optional override for image size
        """
        super().__init__()

        self.tokenizer = tokenizer
        
        # Either use config or individual parameters
        if config is not None:
            self.num_image_tokens = config.num_image_tokens
            self.image_size = config.image_size
        else:
            self.num_image_tokens = num_image_tokens
            self.image_size = image_size

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

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Process images and text for the PaLiGemma model.
        
        Args:
            text: List of text prompts
            images: List of PIL images
            padding: Padding strategy for tokenizer
            truncation: Whether to truncate sequences
            
        Returns:
            dict: Contains 'pixel_values' tensor and tokenizer outputs
        """
        assert len(images) == len(text), f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.num_image_tokens` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.num_image_tokens,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data 