# **SigLIP Vision Transformer Implementation**

This repository contains an implementation of a Vision Transformer (ViT) model following the SigLIP architecture. The model is implemented in a modular way, following Hugging Face’s `transformers` library conventions, making it easy to integrate pretrained weights.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [Configuration (`config.py`)](#configuration-configpy)
  - [Embeddings (`modeling_siglip_vision_embeddings.py`)](#embeddings-modeling_siglip_vision_embeddingspy)
  - [Self-Attention (`modeling_siglip_vision_attention.py`)](#self-attention-modeling_siglip_vision_attentionpy)
  - [MLP (`modeling_siglip_vision_mlp.py`)](#mlp-modeling_siglip_vision_mlppy)
  - [Encoder Layer (`modeling_siglip_vision_encoder_layer.py`)](#encoder-layer-modeling_siglip_vision_encoder_layerpy)
  - [Encoder (`modeling_siglip_vision_encoder.py`)](#encoder-modeling_siglip_vision_encoderpy)
  - [Vision Transformer (`modeling_siglip_vision_transformer.py`)](#vision-transformer-modeling_siglip_vision_transformerpy)
  - [Vision Model (`modeling_siglip_vision_model.py`)](#vision-model-modeling_siglip_vision_modelpy)
  - [Example Script (`run_siglip_vision.py`)](#example-script-run_siglip_visionpy)
- [Results and Outputs](#results-and-outputs)
- [References](#references)

---

## **Introduction**

The SigLIP Vision Transformer is a model that processes images by splitting them into **patches**, converting them into **token embeddings**, and passing them through **self-attention layers** to produce a **contextualized representation** of the image.

This repository implements:
- **Patch embeddings** using a convolutional layer.
- **Multi-head self-attention** for token interactions.
- **Feed-forward layers** to process tokens further.
- **Stacked transformer encoder layers** to build a deep network.
- **Final layer normalization** to stabilize learning.

This implementation follows Hugging Face’s `transformers` design, making it easy to load pretrained weights.

---

## **Project Structure**

```
.
├── config.py                         # Configuration class for hyperparameters
├── modeling_siglip_vision_embeddings.py  # Patch embeddings and positional embeddings
├── modeling_siglip_vision_attention.py   # Multi-head self-attention module
├── modeling_siglip_vision_mlp.py         # Feedforward MLP module
├── modeling_siglip_vision_encoder_layer.py # Transformer encoder layer
├── modeling_siglip_vision_encoder.py      # Transformer encoder (stack of encoder layers)
├── modeling_siglip_vision_transformer.py  # Vision Transformer backbone
├── modeling_siglip_vision_model.py        # Top-level Vision Transformer model
└── run_siglip_vision.py                   # Example script to run the model
```

---

## **Usage**

### **Running the Model**
To test the model on a dummy image:

```bash
python run_siglip_vision.py
```

This script creates a random image of size **(224 × 224)** and passes it through the model.

---

## **Components**

### **Configuration (`config.py`)**
The `SiglipVisionConfig` class stores all hyperparameters for the model.

#### **Key Parameters:**
| Parameter           | Description |
|---------------------|-------------|
| `hidden_size`      | Embedding dimension of each patch (default: 768) |
| `intermediate_size`| Hidden layer size in the MLP (default: 3072) |
| `num_hidden_layers`| Number of transformer encoder layers (default: 12) |
| `num_attention_heads` | Number of attention heads per layer (default: 12) |
| `num_channels` | Number of input channels (default: 3 for RGB) |
| `image_size` | Input image size (default: 224×224) |
| `patch_size` | Patch size for tokenization (default: 16×16) |
| `layer_norm_eps` | Small constant to avoid division by zero (default: 1e-6) |
| `attention_dropout` | Dropout rate for attention (default: 0.0) |

---

## **Results and Outputs**
When running `run_siglip_vision.py`, you should see:

```
Output shape: torch.Size([1, 196, 768])
```

This indicates:
- **1 batch**
- **196 patches** (since \( (224/16)^2 = 196 \))
- **768 embedding dimension per patch**

---

## **References**
- **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)** by Dosovitskiy et al.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** by Vaswani et al.
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**

---

## **Contributing**
Feel free to open **issues** and **pull requests** to improve this implementation.

---

## **License**
This project is licensed under the MIT License.
