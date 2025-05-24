# AION-1 : AstronomIcal Omnimodal Network

Polymathic's Large Omnimodal Model for Astronomy

```python
from aion import AION

model = AION.from_pretrained('/mnt/ceph/users/polymathic/aion/dec24/base')
```

## Installation

AION can be installed in several ways depending on your environment and torch requirements:

### Basic Installation (without torch)

If you already have PyTorch installed or want to manage it separately:

```bash
pip install -e .
```

### Installation with torch

To install AION with PyTorch included:

```bash
pip install -e .[torch]
```

### Development Installation

For development with all dependencies including testing and linting tools:

```bash
pip install -e .[torch,dev]
```

### Custom PyTorch Installation

If you need a specific PyTorch version (e.g., for GPU support), install PyTorch first, then install AION:

```bash
# Install PyTorch with CUDA support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Then install AION without torch
pip install -e .
```

### Conda Environments

For conda users who prefer to manage PyTorch through conda:

```bash
# Install PyTorch via conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install AION without torch
pip install -e .
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.4.0 (optional dependency)
