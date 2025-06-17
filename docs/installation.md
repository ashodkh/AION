# Installation Guide

Quick and straightforward installation guide for AION-1.

## System Requirements

### Hardware Requirements

**Minimum (CPU only)**:
- 16 GB RAM
- 20 GB free storage

**Recommended (GPU)**:
- NVIDIA GPU with 8GB+ VRAM
- 32 GB RAM
- 50 GB free storage

**For Large-Scale Processing**:
- NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A5000+)
- 64GB+ RAM

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Linux, macOS, or Windows

## Installation

### Quick Install (Recommended)

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install AION
pip install aion
```

### Alternative: CPU-only Installation

```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install AION
pip install aion
```

### Development Installation

```bash
git clone https://github.com/polymathic-ai/aion.git
cd aion
pip install -e ".[torch,dev]"
```

## Verification

Test your installation:

```python
import torch
from aion import AION
from aion.codecs import CodecManager

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test model loading (requires internet connection)
try:
    model = AION.from_pretrained('polymathic-ai/aion-base')
    print("✓ AION model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

# Test codec manager
try:
    codec_manager = CodecManager(device='cuda' if torch.cuda.is_available() else 'cpu')
    print("✓ CodecManager initialized successfully")
except Exception as e:
    print(f"✗ CodecManager failed: {e}")
```

## Troubleshooting

### Common Issues

**CUDA out of memory**:
```bash
# Use smaller model or CPU
model = AION.from_pretrained('polymathic-ai/aion-base').to('cpu')
```

**HuggingFace connection issues**:
```bash
# Set up HuggingFace cache directory
export HF_HOME=/path/to/cache
```

**Import errors**:
```bash
# Reinstall with fresh environment
pip uninstall aion torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install aion
```

## Next Steps

Once installed, try the [Tutorial Notebook](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb) or check the [Usage Guide](usage.md) for examples.
