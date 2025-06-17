```{raw} html
<div class="hero-section">
  <div class="hero-background"></div>
  <h1 class="hero-title">AION-1</h1>
  <p class="hero-subtitle">AstronomIcal Omnimodal Network</p>
  <p class="hero-description">Large-Scale Multimodal Foundation Model for Astronomy</p>
  <div class="hero-buttons">
    <a href="#quick-start" class="btn-primary">Get Started →</a>
    <!-- <a href="https://arxiv.org/abs/2406.00000" class="btn-secondary">Read the Paper</a> -->
    <a href="https://colab.research.google.com/github/polymathic-ai/AION/blob/main/notebooks/Tutorial.ipynb" class="btn-secondary">Run on Colab</a>
  </div>
</div>
```

# AION-1 Documentation

## 🌟 Why AION-1?

Trained on over 200 million astronomical objects, AION-1 (AstronomIcal Omnimodal Network) is the first Foundation Model capable of unifying multiband imaging, spectroscopy, and photometry from major ground- and space-based observatories into a single framework.

Compared to traditional machine learning approaches in Astronomy, AION-1 stands out on several points:
- **Enabling Flexible Data Fusion**: Scientists can use any combination of available observations without redesigning their analysis pipeline
- **Enabling Easy Adaptation to Downstream Tasks**: Scientists can adapt AION-1 to new tasks in a matter of minutes and reach SOTA performance
- **Excelling in Low-Data Regimes**: AION-1 achieves competitive results with orders of magnitude less labeled data than supervised approaches
- **Providing Universal Representations**: The learned embeddings capture physically meaningful structure useful across diverse downstream tasks

## 🚀 Quick Start

Getting started with AION-1 is straightforward:

```python
# Minimal end-to-end example
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (LegacySurveyImage, LegacySurveyFluxG,
LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ)

# 1) Load a pre-trained checkpoint (300 M parameters)
model = AION.from_pretrained('polymathic-ai/aion-base').to('cuda').eval()
codec_manager = CodecManager(device='cuda') # Manages codecs for each modality

# 2) Prepare demo inputs (96×96 g,r,i,z cut-out and photometry)
# Create image modality
image = LegacySurveyImage(
   flux=data["legacysurvey_image_flux"],
   bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
)

# Create flux modalities
g = LegacySurveyFluxG(value=data["legacysurvey_FLUX_G"])
r = LegacySurveyFluxR(value=data["legacysurvey_FLUX_R"])
i = LegacySurveyFluxI(value=data["legacysurvey_FLUX_I"])
z = LegacySurveyFluxZ(value=data["legacysurvey_FLUX_Z"])

# Encode input modalities into tokens
tokens = codec_manager.encode(image, g, r, i, z)

# 3) Generate a redshift distribution from these set of inputs
predictions = model(
    tokens,
    target_mask={"tok_z": torch.zeros(batch_size, 1)},
    num_encoder_tokens=600
)
redshift_logits = predictions["tok_z"]  # Shape: [batch, sequence, vocab_size]

# 4) Extract joint embeddings for downstream use
embeddings = model.encode(tokens, num_encoder_tokens=600)  # Shape: [batch, seq_len, hidden_dim]
```

## 📚 Documentation Overview

```{eval-rst}
.. grid:: 2 2 2 4
   :gutter: 3

   .. grid-item-card:: Installation & Setup
      :link: installation.html
      :class-card: doc-card

      Environment setup, dependencies, and configuration

   .. grid-item-card:: Model Specifications
      :link: architecture.html
      :class-card: doc-card

      Deep dive into tokenization, transformers, and trarining data

   .. grid-item-card:: Usage Guide
      :link: usage.html
      :class-card: doc-card

      Tutorials, examples, and best practices

   .. grid-item-card:: API Reference
      :link: api.html
      :class-card: doc-card

      Complete API documentation and method signatures
```

```{toctree}
:hidden:
:maxdepth: 2

installation
architecture
usage
api
```
