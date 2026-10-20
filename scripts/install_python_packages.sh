#!/usr/bin/env bash

# PyTorch (CPU) - versão estável exigida pelo ai-edge-torch
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# TensorFlow nightly (compatível com ai-edge-torch)
pip install tf-nightly-cpu

# AI Edge Torch (versão estável mais recente que suporta quantização PT2E)
pip install ai-edge-torch==0.2.0

# Dependências auxiliares para calibração e processamento de imagem
pip install opencv-python-headless numpy pillow
