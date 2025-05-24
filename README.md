# 🌌 AION-1: AstronomIcal Omnimodal Network

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://github.com/PolymathicAI/AION/actions/workflows/test.yaml/badge.svg)](https://github.com/PolymathicAI/AION/actions/workflows/test.yaml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb)

**Polymathic's Large Omnimodal Model for Astronomy**

[🚀 Quick Start](#-quick-start) • [📦 Installation](#-installation) • [📚 Documentation](#-documentation) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 Overview

AION-1 is a cutting-edge large omnimodal model specifically designed for astronomical applications. It seamlessly integrates multiple modalities of astronomical data to provide comprehensive insights and analysis.

## 🚀 Quick Start

```python
from aion import AION

# Load the pretrained model
model = AION.from_pretrained('aion-base')

# Your astronomical analysis begins here!
```

## 📦 Installation

AION offers flexible installation options to suit your environment and requirements.

### 🔧 Basic Installation

If you already have PyTorch installed or prefer to manage it separately:

```bash
pip install -e .
```

### 🔥 Installation with PyTorch

To install AION with PyTorch included:

```bash
pip install -e .[torch]
```

### 👩‍💻 Development Installation

For contributors and developers:

```bash
pip install -e .[torch,dev]
```

This includes testing frameworks, linting tools, and development dependencies.

### 🎯 Custom PyTorch Installation

For specific PyTorch versions (e.g., CUDA support):

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Then install AION
pip install -e .
```

## 🏗️ Project Structure

```
AION/
├── aion/              # Core library code
├── notebooks/         # Jupyter notebooks and tutorials
│   └── Tutorial.ipynb # Interactive tutorial (Try it on Colab!)
├── tests/            # Test suite
└── README.md         # You are here! 👋
```

## 📚 Documentation

### 🎓 Tutorials

Start with our interactive tutorial:
- **[Open in Google Colab](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb)** - Learn AION basics interactively, no local setup required!

### 🔬 Key Features

- **Multi-modal Integration**: Process various astronomical data types
- **Easy-to-use API**: Simple, intuitive interface for researchers
- **Extensible Framework**: Easy to adapt for specific astronomical tasks

## 🤝 Contributing

We welcome contributions from the astronomical and ML communities!

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies:
   ```bash
   pip install -e .[torch,dev]
   ```
4. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```
5. Run linting:
   ```bash
   ruff check .
   ```
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

AION is developed by [Polymathic AI](https://polymathic-ai.org/), advancing the frontier of AI for scientific discovery.

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/PolymathicAI/AION/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PolymathicAI/AION/discussions)

---

<div align="center">
  <sub>Built with ❤️ for the astronomical community</sub>
</div>
