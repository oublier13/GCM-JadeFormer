# 🌿 JadeFormer: Enhancing Jade Image Retrieval via Self-Supervised Learning with Dynamically Composable Attention

[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review%20at%20The%20Visual%20Computer-blue)](https://link.springer.com/journal/371)
[![License](https://img.shields.io/github/license/your-username/jadeformer?color=2b9348)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/jadeformer)](https://github.com/your-username/jadeformer/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/your-username/jadeformer)](https://github.com/your-username/jadeformer/issues)

> **Jade artifacts embody profound cultural heritage, yet their digital preservation is hindered by complex textures, diverse forms, and limited labeled data.**  
> We introduce **JadeFormer**, a novel self-supervised image retrieval model that achieves **82.7% top-1 accuracy** on a self-constructed jade dataset by integrating dynamic attention, feature fusion, and contrastive learning.

---

## 📌 Overview

This repository contains the official implementation of **JadeFormer**, a self-supervised learning framework designed specifically for **content-based image retrieval (CBIR)** of **Chinese jade artifacts**. Built upon an enhanced **Swin Transformer**, JadeFormer introduces:

- 🧠 **Compositional Window Multi-Head Attention (CW-MHA)**: Dynamically captures fine-grained textures and global morphology.
- 🔗 **Prior Feature Fusion Module (PFFM)**: Enhances sensitivity to jade-specific visual priors via bidirectional cross-attention.
- 📊 **Graph Contrastive Momentum (GCM) Framework**: A novel self-supervised training strategy that models complex negative sample relationships using a dynamic similarity graph.

Our method significantly outperforms existing CBIR approaches while maintaining computational efficiency—making it ideal for cultural heritage digitization and museum-scale applications.

---

## 🧪 Results

| Method          | Top-1 Accuracy (%) |
|-----------------|--------------------|
| ResNet-50       | 64.2               |
| ViT-Base        | 69.8               |
| Swin-Tiny       | 73.5               |
| **JadeFormer (Ours)** | **82.7**           |

> Evaluated on our **self-constructed jade image dataset** (to be released upon publication).

---

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA-compatible GPU (recommended)

### Installation
```bash
git clone https://github.com/your-username/jadeformer.git
cd jadeformer
pip install -r requirements.txt
