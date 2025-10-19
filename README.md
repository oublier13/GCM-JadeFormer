# Enhancing Jade Image Retrieval: A Self-Supervised Learning Approach with Dynamically Composable Attention

[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review%20at%20The%20Visual%20Computer-blue)](https://link.springer.com/journal/371)
[![License](https://img.shields.io/github/license/your-username/jadeformer?color=2b9348)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/jadeformer)](https://github.com/your-username/jadeformer/stargazers)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)

> **Jade artifacts embody profound cultural heritage, yet their digital preservation is hindered by complex textures, diverse forms, and limited labeled data.**  
> We introduce **JadeFormer**, a novel self-supervised image retrieval model that achieves **82.7% top-1 accuracy** on a self-constructed jade dataset by integrating **Compositional Window Multi-Head Attention (CW-MHA)**, **Prior Feature Fusion Module (PFFM)**, and the **Graph Contrastive Momentum (GCM)** framework.

---

## 📌 Overview

This repository implements **JadeFormer**, a hierarchical vision transformer enhanced for **content-based image retrieval (CBIR)** of Chinese jade artifacts. Our method operates in a **self-supervised** setting, eliminating the need for large-scale manual annotations.

### 🔑 Key Innovations
- **Compositional Window Multi-Head Attention (CW-MHA)**: Dynamically composes attention heads to capture fine-grained textures and global morphology.
- **Prior Feature Fusion Module (PFFM)**: Integrates learnable jade-specific prototypes via **bi-directional cross-attention**.
- **Graph Contrastive Momentum (GCM)**: A novel contrastive learning framework that models relationships among hard negative samples using a dynamic similarity graph.
- **Background-invariant training**: Uses synthetic background augmentation to improve robustness in real-world scenarios.

---

## 🧪 Performance

| Method       | Backbone        | Top-1 Acc (%) | Params (M) | FLOPs (G) |
|--------------|------------------|---------------|------------|-----------|
| MoCov3       | ViT-B            | 72.3          | 58.1       | 11.3      |
| FMTH         | SwinTransformer  | 79.5          | 50.0       | 8.7       |
| **Ours (GCM + JadeFormer)** | **JadeFormer**   | **82.7**      | **52.5**   | **9.6**   |

> Evaluated on our **5,000-image jade dataset** with background-augmented training.

---

## 🚀 Installation & Environment

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA-compatible GPU (e.g., RTX 3090)
- Additional libraries (see `requirements.txt`)

### Install Dependencies
```bash
pip install torch torchvision timm einops opencv-python pillow tqdm torch-geometric
