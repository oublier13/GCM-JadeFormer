# A Jade Image Retrieval Method Based on Self-Supervised Learning and Dynamically Composable Attention

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-informational)](#)

> **Jade artifacts embody profound cultural heritage, yet their digital preservation is hindered by complex textures, diverse forms, and limited labeled data.**  
> We introduce **JadeFormer**, a self-supervised image retrieval model that achieves strong performance on a self-constructed jade dataset by integrating **Compositional Window Multi-Head Attention (CW-MHA)**, **Prior Feature Fusion Module (PFFM)**, and the **Graph Contrastive Momentum (GCM)** framework.

---

## 📌 Overview

This repository contains the **Python implementation** of **JadeFormer** for content-based image retrieval (CBIR) of Chinese jade artifacts. The method is trained in a **self-supervised** manner, reducing reliance on large-scale manual labels.

### 🔑 Key Innovations
- **Compositional Window Multi-Head Attention (CW-MHA)**: Dynamically composes attention heads to capture fine-grained textures and global morphology.
- **Prior Feature Fusion Module (PFFM)**: Integrates learnable jade-specific prototypes via bi-directional cross-attention.
- **Graph Contrastive Momentum (GCM)**: A contrastive learning framework that models relationships among hard negatives using a dynamic similarity graph.
- **Background-invariant training**: Uses background augmentation to improve robustness in real-world scenarios.

---

## 🧪 Performance

| Method       | Backbone        | Top-1 Acc (%) | Params (M) | FLOPs (G) |
|--------------|------------------|---------------|------------|-----------|
| MoCov3       | ViT-B            | 72.3          | 58.1       | 11.3      |
| FMTH         | SwinTransformer  | 79.5          | 50.0       | 8.7       |
| **Ours (GCM + JadeFormer)** | **JadeFormer**   | **82.7**      | **52.5**   | **9.6**   |

> Evaluated on a **5,000-image jade dataset** with background-augmented training.

---

## 🚀 Installation & Environment

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA-compatible GPU is recommended (e.g., RTX 3090)

### Install Dependencies
```bash
pip install torch torchvision timm einops opencv-python pillow tqdm torch-geometric

# Optional (if your environment requires it):
# pip install -U "torch-geometric" "torch-scatter" "torch-sparse" "torch-cluster" "torch-spline-conv"
```
